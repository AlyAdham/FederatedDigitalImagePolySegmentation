# Federated Learning for Medical Image Segmentation

This project implements a federated learning framework for medical image segmentation, specifically designed for polyp segmentation in endoscopic images. The system enables collaborative model training across multiple institutions while keeping the data decentralized and private.

## Features

- **Federated Learning**: Train models on decentralized data without sharing raw medical images
- **Polygon-based Segmentation**: Precise segmentation of polyps in endoscopic images
- **PyTorch Implementation**: Built using PyTorch for deep learning
- **Kvasir-SEG Dataset**: Pre-configured to work with the Kvasir-SEG dataset
- **Model Evaluation**: Comprehensive evaluation metrics and visualization tools

## Project Structure

```
.
├── data_loader.py          # Data loading and preprocessing
├── model.py               # Model architecture definitions
├── training.py            # Training loop and federated learning logic
├── test_transnetr_fdg.py   # Testing and evaluation script
├── freq_space_interpolation.py  # Frequency domain operations
├── resnet.py              # Custom ResNet implementation
├── threshold.py           # Post-processing utilities
└── saved_models/          # Directory for trained model checkpoints
```

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- numpy
- matplotlib
- opencv-python

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AlyAdham/FederatedDigitalImagePolySegmentation.git
   cd FederatedDigitalImagePolySegmentation
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Kvasir-SEG dataset and place it in the `Kvasir-SEG` directory.

### Usage

1. **Training**
   ```bash
   python training.py
   ```

2. **Testing**
   ```bash
   python test_transnetr_fdg.py
   ```

## Results

Model performance and visualizations can be found in the `visualize_output` directory after running the evaluation scripts.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Kvasir-SEG dataset for providing the medical imaging data
- PyTorch community for the deep learning framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
