from pathlib import Path
import importlib
import argparse
import sys
from tqdm import trange
from flygym import Camera
from cobar_miniproject import levels
from cobar_miniproject.cobar_fly import CobarFly
from flygym import Camera, SingleFlySimulation
from flygym.arena import FlatTerrain


def run_simulation(
    submission_dir,
    level,
    seed,
    debug,
    max_steps,
    output_dir="outputs",
    progress=True,
):
    sys.path.append(str(submission_dir.parent))
    module = importlib.import_module(submission_dir.name)
    controller = module.controller.Controller()
    timestep = 1e-4

    fly = CobarFly(
        debug=debug,
        enable_vision=True,
        render_raw_vision=True,
    )

    if level <= -1:
        level_arena = FlatTerrain()
    elif level <= 1:
        # levels 0 and 1 don't need the timestep
        level_arena = levels[level](fly=fly, seed=seed)
    else:
        # levels 2-4 need the timestep
        level_arena = levels[level](fly=fly, timestep=timestep, seed=seed)
    
    cam_params = {"pos": (0, 0, 80)}

    cam = Camera(
        attachment_point=level_arena.root_element.worldbody,
        camera_name="camera_top_zoomout",
        targeted_fly_names=[fly.name],
        camera_parameters=cam_params,
        play_speed=0.2,
    )

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=level_arena,
    )

    # run cpg simulation
    obs, info = sim.reset()
    obs_hist = []
    info_hist = []

    if progress:
        step_range = trange(max_steps)
    else:
        step_range = range(max_steps)

    for i in step_range:
        # Get observations
        obs, reward, terminated, truncated, info = sim.step(controller.get_actions(obs))
        sim.render()
        if controller.done_level(obs):
            # finish the path integration level
            break
            
        obs_ = obs.copy()
        if not obs_["vision_updated"]:
            if "vision" in obs_:
                del obs_["vision"]
            if "raw_vision" in obs_:
                del obs_["raw_vision"]
        if "raw_vision" in info:
            del info["raw_vision"]
        obs_hist.append(obs_)
        info_hist.append(info)

        if hasattr(controller, "quit") and controller.quit:
            print("Simulation terminated by user.")
            break
        if hasattr(level_arena, "quit") and level_arena.quit:
            print("Target reached. Simulation terminated.")
            break

    # Save video
    save_path = Path(output_dir) / f"level{level}_seed{seed}.mp4"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cam.save_video(save_path, stabilization_time=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fly simulation.")
    parser.add_argument(
        "submission_dir",
        type=Path,
        help="Path to the submission directory containing the controller module.",
    )
    parser.add_argument(
        "--level",
        type=int,
        help="Simulation level to run (e.g., -1 for FlatTerrain, 0-4 for specific levels).",
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for the simulation.",
        default=0,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum number of steps to run the simulation.",
        default=10000,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for the simulation.",
        default=False,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save the simulation outputs (default: 'outputs').",
        default="outputs",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar during simulation.",
    )
    args = parser.parse_args()

    run_simulation(
        submission_dir=args.submission_dir,
        level=args.level,
        seed=args.seed,
        debug=args.debug,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        progress=args.progress,
    )
