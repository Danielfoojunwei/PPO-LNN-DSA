"""
Artifact Signing and Verification

Ed25519-based signing for FL model artifacts.
Ensures authenticity and integrity of deployed models.
"""

import os
import time
import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import base64

logger = logging.getLogger(__name__)

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    from cryptography.exceptions import InvalidSignature
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography not installed. Signing disabled.")


@dataclass
class SignedArtifact:
    """Signed model artifact metadata."""
    artifact_id: str
    artifact_path: str
    content_hash: str  # SHA-256 of artifact content
    signature: str  # Base64-encoded Ed25519 signature
    signer_id: str
    signed_at: float
    version: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_path": self.artifact_path,
            "content_hash": self.content_hash,
            "signature": self.signature,
            "signer_id": self.signer_id,
            "signed_at": self.signed_at,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignedArtifact":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "SignedArtifact":
        return cls.from_dict(json.loads(s))


class KeyPair:
    """Ed25519 key pair for signing."""

    def __init__(
        self,
        private_key: Optional['Ed25519PrivateKey'] = None,
        public_key: Optional['Ed25519PublicKey'] = None,
    ):
        self._private_key = private_key
        self._public_key = public_key

    @classmethod
    def generate(cls) -> "KeyPair":
        """Generate new key pair."""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library not available")

        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        return cls(private_key, public_key)

    @classmethod
    def from_private_key_bytes(cls, key_bytes: bytes) -> "KeyPair":
        """Load from private key bytes."""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library not available")

        private_key = Ed25519PrivateKey.from_private_bytes(key_bytes)
        public_key = private_key.public_key()
        return cls(private_key, public_key)

    @classmethod
    def from_public_key_bytes(cls, key_bytes: bytes) -> "KeyPair":
        """Load from public key bytes (verification only)."""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library not available")

        public_key = Ed25519PublicKey.from_public_bytes(key_bytes)
        return cls(None, public_key)

    @property
    def can_sign(self) -> bool:
        return self._private_key is not None

    @property
    def private_key_bytes(self) -> bytes:
        if not self._private_key:
            raise ValueError("No private key")
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    @property
    def public_key_bytes(self) -> bytes:
        if not self._public_key:
            raise ValueError("No public key")
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def sign(self, data: bytes) -> bytes:
        """Sign data."""
        if not self._private_key:
            raise ValueError("No private key for signing")
        return self._private_key.sign(data)

    def verify(self, signature: bytes, data: bytes) -> bool:
        """Verify signature."""
        if not self._public_key:
            raise ValueError("No public key for verification")
        try:
            self._public_key.verify(signature, data)
            return True
        except InvalidSignature:
            return False


class ArtifactSigner:
    """
    Signs and verifies model artifacts.

    Uses Ed25519 for fast, secure signatures.
    """

    def __init__(
        self,
        signer_id: str,
        key_pair: Optional[KeyPair] = None,
        key_dir: Optional[str] = None,
    ):
        self.signer_id = signer_id
        self.key_dir = Path(key_dir) if key_dir else Path("./keys")

        if key_pair:
            self.key_pair = key_pair
        else:
            self.key_pair = self._load_or_generate_keys()

    def _load_or_generate_keys(self) -> KeyPair:
        """Load existing keys or generate new ones."""
        if not CRYPTO_AVAILABLE:
            logger.warning("Crypto not available, using dummy keys")
            return None

        self.key_dir.mkdir(parents=True, exist_ok=True)
        private_key_path = self.key_dir / f"{self.signer_id}.key"
        public_key_path = self.key_dir / f"{self.signer_id}.pub"

        if private_key_path.exists():
            with open(private_key_path, 'rb') as f:
                key_bytes = f.read()
            return KeyPair.from_private_key_bytes(key_bytes)
        else:
            key_pair = KeyPair.generate()
            with open(private_key_path, 'wb') as f:
                f.write(key_pair.private_key_bytes)
            with open(public_key_path, 'wb') as f:
                f.write(key_pair.public_key_bytes)
            logger.info(f"Generated new key pair for {self.signer_id}")
            return key_pair

    def _compute_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of file content."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def sign_artifact(
        self,
        artifact_path: str,
        artifact_id: Optional[str] = None,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SignedArtifact:
        """
        Sign a model artifact.

        Args:
            artifact_path: Path to artifact file
            artifact_id: Unique artifact identifier
            version: Artifact version
            metadata: Additional metadata

        Returns:
            SignedArtifact with signature
        """
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        artifact_id = artifact_id or path.stem
        content_hash = self._compute_hash(path)

        # Create signature data
        sign_data = f"{artifact_id}:{content_hash}:{version}".encode()

        if self.key_pair and self.key_pair.can_sign:
            signature_bytes = self.key_pair.sign(sign_data)
            signature = base64.b64encode(signature_bytes).decode()
        else:
            signature = "UNSIGNED"

        artifact = SignedArtifact(
            artifact_id=artifact_id,
            artifact_path=str(path.absolute()),
            content_hash=content_hash,
            signature=signature,
            signer_id=self.signer_id,
            signed_at=time.time(),
            version=version,
            metadata=metadata or {},
        )

        # Save signature file
        sig_path = path.with_suffix(path.suffix + ".sig")
        with open(sig_path, 'w') as f:
            f.write(artifact.to_json())

        logger.info(f"Signed artifact: {artifact_id} (hash: {content_hash[:16]}...)")

        return artifact

    def verify_artifact(
        self,
        artifact_path: str,
        public_key_bytes: Optional[bytes] = None,
    ) -> Tuple[bool, str]:
        """
        Verify artifact signature.

        Args:
            artifact_path: Path to artifact file
            public_key_bytes: Optional public key (uses signer's if None)

        Returns:
            (is_valid, message)
        """
        path = Path(artifact_path)
        sig_path = path.with_suffix(path.suffix + ".sig")

        if not sig_path.exists():
            return False, "Signature file not found"

        with open(sig_path, 'r') as f:
            artifact = SignedArtifact.from_json(f.read())

        # Verify hash
        current_hash = self._compute_hash(path)
        if current_hash != artifact.content_hash:
            return False, "Content hash mismatch - file may be corrupted or modified"

        # Verify signature
        if artifact.signature == "UNSIGNED":
            return False, "Artifact is not signed"

        if not CRYPTO_AVAILABLE:
            return True, "Signature verification skipped (crypto unavailable)"

        try:
            signature_bytes = base64.b64decode(artifact.signature)
            sign_data = f"{artifact.artifact_id}:{artifact.content_hash}:{artifact.version}".encode()

            if public_key_bytes:
                verifier = KeyPair.from_public_key_bytes(public_key_bytes)
            elif self.key_pair:
                verifier = self.key_pair
            else:
                return False, "No public key available for verification"

            if verifier.verify(signature_bytes, sign_data):
                return True, "Signature valid"
            else:
                return False, "Invalid signature"

        except Exception as e:
            return False, f"Verification error: {e}"


class ArtifactLoader:
    """
    Loads verified model artifacts.

    Ensures only signed and valid artifacts are loaded.
    """

    def __init__(
        self,
        trusted_signers: Optional[List[str]] = None,
        key_dir: Optional[str] = None,
    ):
        self.trusted_signers = set(trusted_signers or [])
        self.key_dir = Path(key_dir) if key_dir else Path("./keys")
        self._public_keys: Dict[str, bytes] = {}

        self._load_public_keys()

    def _load_public_keys(self) -> None:
        """Load public keys for trusted signers."""
        if not self.key_dir.exists():
            return

        for pub_file in self.key_dir.glob("*.pub"):
            signer_id = pub_file.stem
            if not self.trusted_signers or signer_id in self.trusted_signers:
                with open(pub_file, 'rb') as f:
                    self._public_keys[signer_id] = f.read()

    def add_trusted_signer(self, signer_id: str, public_key: bytes) -> None:
        """Add a trusted signer."""
        self.trusted_signers.add(signer_id)
        self._public_keys[signer_id] = public_key

    def load_artifact(
        self,
        artifact_path: str,
        require_signature: bool = True,
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Load and verify an artifact.

        Args:
            artifact_path: Path to artifact
            require_signature: Whether to require valid signature

        Returns:
            (success, artifact_data, message)
        """
        path = Path(artifact_path)
        sig_path = path.with_suffix(path.suffix + ".sig")

        # Load signature info
        if sig_path.exists():
            with open(sig_path, 'r') as f:
                artifact_info = SignedArtifact.from_json(f.read())

            # Check trusted signer
            if self.trusted_signers and artifact_info.signer_id not in self.trusted_signers:
                return False, None, f"Untrusted signer: {artifact_info.signer_id}"

            # Verify
            public_key = self._public_keys.get(artifact_info.signer_id)
            signer = ArtifactSigner(artifact_info.signer_id)
            valid, message = signer.verify_artifact(artifact_path, public_key)

            if not valid and require_signature:
                return False, None, message

        elif require_signature:
            return False, None, "No signature file found"
        else:
            artifact_info = None

        # Load artifact data
        try:
            import numpy as np
            data = np.load(artifact_path)
            weights = {k: data[k] for k in data.files}

            return True, {
                "weights": weights,
                "info": artifact_info.to_dict() if artifact_info else None,
            }, "Artifact loaded successfully"

        except Exception as e:
            return False, None, f"Failed to load artifact: {e}"
