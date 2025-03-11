# UrbanVideo-Bench: Benchmarking Vision-Language Models on Embodied Intelligence with Video Data in Urban Spaces

The benchmark is designed to evaluate whether video-large language models (Video-LLMs) can naturally process continuous first-person visual observations like humans, enabling recall, perception, reasoning, and navigation.

- **Arxiv**: https://arxiv.org/pdf/2503.06157
- **Project**: https://embodiedcity.github.io/UrbanVideo-Bench/
- **Dataset**: https://huggingface.co/datasets/EmbodiedCity/UrbanVideo-Bench

## News
✅ Dataset Upload

⬜ Dataset generation code (To be completed)

⬜ Code for running the benchmark with Video-LLMs (To be completed)

## Dataset generation
The pipeline includes four steps: video curation, MCQ generation, blind filtering, and human refinement. 
The dataset statistics are shown in the following figure b-f.
![UrbanVideo-Bench.code](flowchart.png)

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