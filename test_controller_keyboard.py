from pathlib import Path
import importlib
import argparse
import sys
import cv2
import dm_control.rl.control
import numpy as np
import pandas as pd
from tqdm import trange
from flygym import Camera, YawOnlyCamera
from cobar_miniproject import levels
from cobar_miniproject.cobar_fly import CobarFly
from flygym import Camera, SingleFlySimulation

from cobar_miniproject.keyboard_controller import KeyBoardController
from cobar_miniproject.vision import get_fly_vision, render_image_with_vision


def run_simulation(
    submission_dir,
    level,
    seed,
    max_steps,
    output_dir="outputs",
    progress=True,
):
    timestep = 1e-4
    controller = KeyBoardController(timestep=timestep)

    fly = CobarFly(
        debug=False,
        enable_vision=True,
        render_raw_vision=True,
    )

    if 0 > level > 4:
        raise ValueError("level should be between 0 and 4.")
    elif level <= 1:
        # levels 0 and 1 don't need the timestep
        level_arena = levels[level](fly=fly, seed=seed)
    else:
        # levels 2-4 need the timestep
        level_arena = levels[level](fly=fly, timestep=timestep, seed=seed)
    
    # cam_params = {"pos": (0, 0, 80)}

    # cam = Camera(
    #     attachment_point=level_arena.root_element.worldbody,
    #     camera_name="camera_top_zoomout",
    #     targeted_fly_names=[fly.name],
    #     camera_parameters=cam_params,
    #     play_speed=0.2,
    # )

    cam = YawOnlyCamera(
        attachment_point=fly.model.worldbody,
        camera_name="camera_back_track_game",
        targeted_fly_names=[fly.name],
        play_speed=0.2,
    )

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=level_arena,
    )

    obs, info = sim.reset()

    if progress:
        step_range = trange(max_steps)
    else:
        step_range = range(max_steps)

    for i in step_range:
        # Get observations
        try:
            obs, reward, terminated, truncated, info = sim.step(
                controller.get_actions(obs)
            )
        except dm_control.rl.control.PhysicsError:
            result = (0, "physics error - probably fly got hit by the ball")
        
        rendered_img = sim.render()[0]
        if rendered_img is not None:
            rendered_img = render_image_with_vision(
                rendered_img,
                get_fly_vision(fly),
                obs["odor_intensity"],
            )
            rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
            cv2.imshow("Simulation", rendered_img)
            cv2.waitKey(1)

        if terminated or truncated:
            result = (0, "simulated terminated by error")

        if controller.done_level(obs):
            if level == 4:
                if level_arena.state != "returning":
                    result = (
                        0,
                        "controller terminated during path integration before getting odour",
                    )
                    break
                # finish the path integration level
                distance_from_origin = np.linalg.norm(
                    fly.last_obs["pos"][:2]
                )
                # if within 3 mm - full points
                # if further than 20 mm (which is ~65% of the initial odour distance of 29-31) - no points
                # else linearly interpolate points from distance
                min_scoring_distance = 3
                max_scoring_distance = 20
                
                if distance_from_origin <= min_scoring_distance:
                    points = 1
                elif distance_from_origin > max_scoring_distance:
                    points = 0
                else:
                    points = 1 + (min_scoring_distance - distance_from_origin) / (
                        max_scoring_distance - min_scoring_distance
                    )

                result = (points, "controller terminated during path integration")
            else:
                result = (0, "controller wrongly terminated the level")
            break

        if hasattr(level_arena, "quit") and level_arena.quit:
            result = (1, "reached odour")
            break
    else:
        # simulation took too long - didn't break the loop
        result = (0, "timed out")

    # Save video
    save_path = Path(output_dir) / f"level{level}_seed{seed}.mp4"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cam.save_video(save_path, stabilization_time=0)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fly simulation.")
    parser.add_argument(
        "submission_dir",
        type=Path,
        help="Path to the submission directory containing the controller module.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        help="random seeds to test your controller on - pass as a comma delimited list. eg 1,2,3,4,10",
        default="45",
    )
    parser.add_argument(
        "--levels",
        type=str,
        help="levels to test your controller on - pass as a comma delimited list. eg 0,1,4",
        default="0,1,2,3,4",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save the simulation outputs (default: 'outputs').",
        default="outputs",
    )
    args = parser.parse_args()

    levels_to_test = [int(levels) for levels in args.levels.split(",")]
    seeds_to_test = [int(seed) for seed in args.seeds.split(",")]

    print(f"Testing levels {levels_to_test} with random seeds {seeds_to_test}")

    results = {}

    for level in levels_to_test:
        print(f"testing level {level}")
        results[level] = {}

        for seed in seeds_to_test:
            points, reason = run_simulation(
                submission_dir=args.submission_dir,
                level=level,
                seed=seed,
                max_steps=100_000 if level < 4 else 200_000,
                output_dir=args.output_dir,
                progress=True,
            )

            results[level][seed] = (points, reason)
            print(f"Level {level}, seed {seed}: {points} points - {reason}")
    
    print(f"Total points: {sum([seed_result[0] for level_results in results.values() for seed_result in level_results.values()]):.2f}/{len(levels_to_test)*len(seeds_to_test)}")

    print("All results:")
    print(results)

    pd.DataFrame(results).to_csv("results.csv")
