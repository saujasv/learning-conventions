# Training models to form conventions in repeated reference games

For all experiments, ensure images are downloaded, and the base path of the images directory is set in the environment variable `$IMAGE_BASE_PATH`. For COCO, this is a folder containing `train2014` and `val2014` subfolders, which in turn contain images. For tangrams, it is a folder with JPEG files of the black tangram images from the [Kilogram paper](https://github.com/lil-lab/kilogram).

To simulate games, run
```
python -m fire training/simulate_games.py run_sampling $CONFIG_FILE
```

The `CONFIG_FILE` is a JSON file. Examples can be found in `e2e_configs/`.

To train models, run 
```
accelerate launch -m fire training/dpo.py train --config ${CONFIG_FILE}
```

The `CONFIG_FILE` is a YAML file. Examples can be found in `e2e_configs/`.