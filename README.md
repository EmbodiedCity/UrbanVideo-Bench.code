# UrbanVideo-Bench: Benchmarking Vision-Language Models on Embodied Intelligence with Video Data in Urban Spaces

The benchmark is designed to evaluate whether video-large language models (Video-LLMs) can naturally process continuous first-person visual observations like humans, enabling recall, perception, reasoning, and navigation.

- **Arxiv**: https://arxiv.org/pdf/2503.06157
- **Project**: https://embodiedcity.github.io/UrbanVideo-Bench/
- **Dataset**: https://huggingface.co/datasets/EmbodiedCity/UrbanVideo-Bench

## News
✅ Dataset Upload

⬜ Dataset generation code (To be completed)

✅ Example code for running the benchmark with Video-LLMs

## Dataset generation
The pipeline includes four steps: video curation, MCQ generation, blind filtering, and human refinement. 
The dataset statistics are shown in the following figure b-f.
![UrbanVideo-Bench.code](flowchart.png)

## Example code for running the benchmark
### Data Preparation

To get started, download the dataset from [Hugging Face](https://huggingface.co/datasets/EmbodiedCity/UrbanVideo-Bench) 
and place it in the `dataset` folder within the project directory. 
After downloading, ensure the folder structure matches the one described below.
```
UrbanVideo-Bench.code/
├── dataset/
│   ├── videos/          # Contains video files used as input for the model
│   ├── MCQ.parquet      # Contains multiple-choice questions
│   └── ...
├── run.py               # Script for running the model and generating predictions
├── eval.py              # Script for evaluating the model's predictions
├── README.md            # Documentation for the project
└── ...                  # Other potential files or subdirectories
```

### Running

1. Set the model name in `run.py`:
   ```python
   model = "your_model_name"
   ```

2. Configure OpenAI API credentials:
   ```python
   client = OpenAI(
       api_key='your_api_key',
       base_url='your_base_url'
   )
   ```

3. Run the script:
   ```bash
   python run.py
   ```
   Results will be saved to `result/%s_output.csv`.


### Evaluation

1. Modify the file path in `eval.py` to match the output file from `run.py`:
   ```python
   file_path = 'result/gpt-4o_output.csv'  # Replace with your output file path
   ```

2. Run the script:
   ```bash
   python eval.py
   ```

3. The script compares predictions to ground truth and calculates accuracy. Results are saved to: `result/%s_acc.xlsx`


## Citation
If you use this project in your research, please cite the following paper:

```
@misc{zhao2025urbanvideobench,
      title={UrbanVideo-Bench: Benchmarking Vision-Language Models on Embodied Intelligence with Video Data in Urban Spaces}, 
      author={Baining Zhao and Jianjie Fang and Zichao Dai and Ziyou Wang and Jirong Zha and Weichen Zhang and Chen Gao and Yue Wang and Jinqiang Cui and Xinlei Chen and Yong Li},
      year={2025},
      eprint={2503.06157},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.06157}, 
}
```