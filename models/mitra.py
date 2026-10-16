"""Mitra (AutoGluon's Tab2D) as a frozen in-context classifier.

``MitraClassifier`` fine-tunes by default, holds out 20% of the training data
for validation and randomly subsamples contexts above 8192 rows on every
prediction.  The other tabular foundation models in this repo are audited as
frozen models prompted with their *whole* training set, so this wrapper does
the same for Mitra: ``fit`` only fits the preprocessor and stores the context
(no gradient step, no validation split), and prediction attends to every
context row unless ``max_context`` caps it.

It also gives Mitra what the pipeline expects from a model: original-valued
``classes_``, ``predict`` in the original label space, ``to(device)``, tensor
inputs, indexed devices such as ``cuda:1`` and pickles without the pretrained
weights (they never change, so they are reloaded from the Hugging Face cache).

The attention attack and the defenses patch ``F.scaled_dot_product_attention``
and recognise calls by their shape, so the wrapper keeps three invariants:
flash attention is off, every query chunk is shorter than the context
(query-to-context row attention is the only call with q_len < k_len ==
``context_len``), and the context is always presented in ``context_order()``.
"""

import contextlib
import copy

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from autogluon.tabular.models.mitra._internal.core.trainer_finetune import TrainerFinetune
from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D
from autogluon.tabular.models.mitra.sklearn_interface import (
    DEFAULT_CLASSES,
    DEFAULT_CLS_MODEL,
    MitraClassifier,
)

MITRA_MAX_CLASSES = DEFAULT_CLASSES  # output size of the pretrained classifier head


def _to_numpy(X) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    elif hasattr(X, "values"):
        X = X.values
    return np.asarray(X, dtype=np.float32)


class MitraICLClassifier(MitraClassifier):
    """Mitra prompted with its training set, without fine-tuning.

    max_context: cap on the context rows attended to per prediction (a random
        subset is drawn for every query chunk when the context is larger).
        None uses the whole training set.
    """

    def __init__(self, device="cuda", n_estimators=1, max_context=None,
                 hf_model=DEFAULT_CLS_MODEL, seed=0):
        super().__init__(
            n_estimators=n_estimators,
            device=device,
            fine_tune=False,
            hf_model=hf_model,
            seed=seed,
            verbose=False,
        )
        self.max_context = max_context

    # Mitra passes its device string to torch.autocast, which only accepts a
    # device *type*.  Trainers therefore get "cuda"/"cpu" and an indexed device
    # is selected by making it the current CUDA device around every call.
    @property
    def _device_type(self) -> str:
        return torch.device(self.device).type

    @contextlib.contextmanager
    def _on_device(self):
        index = torch.device(self.device).index
        if self._device_type == "cuda" and index is not None:
            with torch.cuda.device(index):
                yield
        else:
            yield

    def _load_model(self):
        with self._on_device():
            model = Tab2D.from_pretrained(self.hf_model, device=self._device_type)
        for module in model.modules():
            if getattr(module, "use_flash_attn", False):
                module.use_flash_attn = False   # exposes no attention weights to hooks
        return model.eval()

    @property
    def context_len(self) -> int:
        """Context rows attended to per prediction."""
        return min(len(self.X), int(self.max_context or len(self.X)))

    @property
    def query_chunk_size(self) -> int:
        """Query rows per forward pass: Mitra's own chunk size, kept below the context."""
        max_query = self.trainers[0].cfg.hyperparams["max_samples_query"]
        return max(1, min(int(max_query), self.context_len - 1))

    def context_order(self) -> np.ndarray:
        """Indices into the fitted context, in the order its rows reach the model."""
        return np.random.RandomState(self.seed).choice(
            len(self.X), size=self.context_len, replace=False)

    def _ensure_models(self):
        for trainer in self.trainers:
            if trainer.model is None:
                trainer.model = self._load_model()
            trainer.device = self._device_type

    def fit(self, X, y):
        X = _to_numpy(X)
        encoder = LabelEncoder()
        y = encoder.fit_transform(np.asarray(y).reshape(-1)).astype(np.int64)
        if len(encoder.classes_) > MITRA_MAX_CLASSES:
            raise ValueError(
                f"Mitra supports at most {MITRA_MAX_CLASSES} classes, got {len(encoder.classes_)}."
            )
        self.classes_ = encoder.classes_
        self.X, self.y = X, y

        cfg, _ = self._create_config(self.task, DEFAULT_CLASSES)
        cfg.device = self._device_type
        cfg.hyperparams["max_samples_support"] = int(self.max_context or len(X))
        rng = np.random.RandomState(self.seed)

        self.trainers = []
        for _ in range(self.n_estimators):
            trainer = TrainerFinetune(cfg, self._load_model(), n_classes=DEFAULT_CLASSES,
                                      device=self._device_type, rng=rng, verbose=False)
            trainer.preprocessor.fit(X.copy(), y)
            trainer.post_fit_optimize()
            self.trainers.append(trainer)
        return self

    def predict_proba(self, X):
        self._ensure_models()
        X = _to_numpy(X)
        step = self.query_chunk_size
        probs = []
        with self._on_device():
            for start in range(0, len(X), step):
                # Mitra shuffles (and, above max_context, subsamples) the context with
                # the trainer's RNG for every chunk; restart it so that every query
                # sees the context in context_order().
                for trainer in self.trainers:
                    trainer.rng = np.random.RandomState(self.seed)
                probs.append(super().predict_proba(X[start:start + step]))
        return np.concatenate(probs)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def to(self, device):
        """Move the pretrained weights; the context itself stays in host memory."""
        self.device = str(device)
        self._ensure_models()
        with self._on_device():
            for trainer in self.trainers:
                trainer.model = trainer.model.to(self._device_type)
        return self

    def __getstate__(self):
        state = dict(super().__getstate__())  # sklearn may hand back __dict__ itself
        trainers = []
        for trainer in self.trainers:
            trainer = copy.copy(trainer)
            trainer.model = None  # frozen pretrained weights, reloaded on first use
            trainers.append(trainer)
        state["trainers"] = trainers
        return state
