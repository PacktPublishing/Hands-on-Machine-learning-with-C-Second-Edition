import optuna
import subprocess
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PolyFit")
    parser.add_argument(
        "binary_path", type=str, help="path to the optimization process binary"
    )
    args = parser.parse_args()

    search_space = {
        "learning_rate": [0.01, 0.025, 0.045],
        "polynomial_degree": [8, 14, 16],
        "batch_size": [16, 32, 64],
    }
    study = optuna.create_study(
        study_name="PolyFit",
        direction="minimize",
        sampler=optuna.samplers.GridSampler(search_space),
    )

    def objective(trial: optuna.trial.Trial):
        lr = trial.suggest_float("learning_rate", low=0.01, high=0.05)
        d = trial.suggest_int("polynomial_degree", low=8, high=16)
        bs = trial.suggest_int("batch_size", low=16, high=64)

        print(
            f"Start training with: learning_rate=f{lr}, polynomial_degree={d}, batch_size={bs}\n"
        )

        result = subprocess.run(
            [args.binary_path, str(d), str(lr), str(bs)], stdout=subprocess.PIPE
        )
        mse = float(result.stdout)
        return mse

    study.optimize(objective)
    print(f"Best value: {study.best_value} (params: {study.best_params})\n")
