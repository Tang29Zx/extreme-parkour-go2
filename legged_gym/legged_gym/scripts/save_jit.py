import json
import os, sys
from pathlib import Path
from statistics import mode
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import Actor, StateHistoryEncoder, get_activation, ActorCriticRMA
from rsl_rl.modules.estimator import Estimator
from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
import argparse
import code
import shutil

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path, checkpoint

class HardwareVisionNN(nn.Module):
    def __init__(self,  num_prop,
                        num_scan,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        tanh,
                        actor_hidden_dims=[512, 256, 128],
                        scan_encoder_dims=[128, 64, 32],
                        depth_encoder_hidden_dim=512,
                        activation='elu',
                        priv_encoder_dims=[64, 20]
                        ):
        super(HardwareVisionNN, self).__init__()

        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        num_obs = num_prop + num_scan + num_hist*num_prop + num_priv_latent + num_priv_explicit
        self.num_obs = num_obs
        activation = get_activation(activation)
        
        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=tanh)

        self.estimator = Estimator(input_dim=num_prop, output_dim=num_priv_explicit, hidden_dims=[128, 64])
        
    def forward(self, obs, depth_latent):
        obs[:, self.num_prop+self.num_scan : self.num_prop+self.num_scan+self.num_priv_explicit] = self.estimator(obs[:, :self.num_prop])
        return self.actor(obs, hist_encoding=True, eval=False, scandots_latent=depth_latent)
        # return obs, depth_latent

def _load_manifest_env_cfg(load_run):
    manifest_path = Path(load_run) / "run_manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")

    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if not records or "env_cfg" not in records[-1]:
        raise ValueError(f"Run manifest does not contain env_cfg: {manifest_path}")
    return records[-1]["env_cfg"]


def play(args):
    if args.load_dir:
        load_run = os.path.abspath(args.load_dir)
    else:
        legged_gym_root = Path(__file__).resolve().parents[2]
        load_run = str(legged_gym_root / "logs" / args.proj_name / args.exptid)
    checkpoint = args.checkpoint

    n_priv_explicit = 3 + 3 + 3
    n_priv_latent = 4 + 1 + 12 +12
    num_scan = 132
    num_actions = 12

    # depth_buffer_len = 2
    depth_resized = (87, 58)
    
    n_proprio = 3 + 2 + 3 + 4 + 36 + 4 +1
    history_len = 10

    device = torch.device('cpu')
    policy = HardwareVisionNN(n_proprio, num_scan, n_priv_latent, n_priv_explicit, history_len, num_actions, args.tanh).to(device)
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=checkpoint)
    load_run = os.path.dirname(load_path)
    print(f"Loading model from: {load_path}")
    ac_state_dict = torch.load(load_path, map_location=device, weights_only=True)
    # policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)
    policy.actor.load_state_dict(ac_state_dict['depth_actor_state_dict'], strict=True)
    policy.estimator.load_state_dict(ac_state_dict['estimator_state_dict'])
    
    policy = policy.to(device)#.cpu()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(load_run) / "traced"
    if args.output_dir:
        base_path = output_dir / "base_jit.pt"
        vision_path = output_dir / "vision_weight.pt"
    else:
        base_path = output_dir / f"{args.exptid}-{checkpoint}-base_jit.pt"
        vision_path = output_dir / f"{args.exptid}-{checkpoint}-vision_weight.pt"
    config_path = output_dir / "config.json"
    output_paths = [base_path, vision_path]
    if args.export_config:
        output_paths.append(config_path)
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not args.force:
        raise FileExistsError(
            "Refusing to overwrite existing export files: "
            + ", ".join(str(path) for path in existing_paths)
        )

    env_cfg = _load_manifest_env_cfg(load_run) if args.export_config else None
    state_dict = {'depth_encoder_state_dict': ac_state_dict['depth_encoder_state_dict']}

    # Save the traced actor
    policy.eval()
    with torch.no_grad(): 
        num_envs = 1
        
        obs_input = torch.ones(num_envs, n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len*n_proprio, device=device)
        depth_latent = torch.ones(1, 32, device=device)
        test = policy(obs_input, depth_latent)
        
        traced_policy = torch.jit.trace(policy, (obs_input, depth_latent))
        
        # traced_policy = torch.jit.script(policy)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict, vision_path)
        traced_policy.save(str(base_path))
        if args.export_config:
            config_path.write_text(json.dumps(env_cfg, indent=4, sort_keys=True) + "\n")
        print("Saved base JIT at", base_path)
        print("Saved vision weights at", vision_path)
        if args.export_config:
            print("Saved environment config at", config_path)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exptid', type=str)
    parser.add_argument('--proj_name', type=str, default='parkour_new')
    parser.add_argument('--load-dir', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--export-config', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--tanh', action='store_true')
    args = parser.parse_args()
    play(args)
