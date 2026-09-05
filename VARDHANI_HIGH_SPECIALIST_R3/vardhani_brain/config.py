from dataclasses import dataclass, asdict
import json

@dataclass
class BrainConfig:
    name: str = "vardhani-specialist"
    input_dim: int = 128
    lower_dim: int = 96
    context15_dim: int = 32
    hidden_dim: int = 768
    layers: int = 12
    heads: int = 12
    ff_mult: int = 4
    experts: int = 10
    expert_hidden: int = 1536
    seq_len: int = 256
    latent_dim: int = 256
    regimes: int = 16
    actions: int = 3  # PE / WAIT / CE
    horizons: tuple = (1,3,5,10,15)
    dropout: float = 0.08
    episodic_dim: int = 256
    option_dim: int = 32
    model_asset_budget_bytes: int = 600_000_000
    memory_asset_budget_bytes: int = 250_000_000
    total_android_ceiling_bytes: int = 1_073_741_824

    @classmethod
    def load(cls, path):
        d=json.load(open(path))
        if "horizons" in d: d["horizons"]=tuple(d["horizons"])
        return cls(**d)

    def save(self,path):
        d=asdict(self); d["horizons"]=list(self.horizons)
        json.dump(d,open(path,"w"),indent=2)
