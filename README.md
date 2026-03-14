# Autonomous Vehicle Perception Stack

This project implements a simplified camera-based perception pipeline similar to those used in autonomous driving systems. The system combines multiple computer vision and deep learning modules to understand urban driving environments using the Cityscapes dataset.

## Overview

Autonomous vehicles rely on perception systems to interpret their surroundings. This project demonstrates how several perception tasks can be integrated to produce a more comprehensive understanding of road scenes.

The pipeline processes camera images and performs:

- Object Detection
- Semantic Segmentation
- Lane Detection
- Distance Estimation
- Collision Risk Detection

Together these modules simulate a simplified perception stack used in modern self-driving systems.

## Features

### Object Detection
Uses YOLOv8 to detect dynamic road actors including:

- Cars
- Trucks
- Buses
- Pedestrians
- Traffic Lights
- Motorcycles

Bounding boxes and confidence scores are displayed for each detected object.

### Semantic Segmentation
Uses SegFormer (trained on Cityscapes) to classify each pixel of the scene into road-related categories such as:

- Road
- Sidewalk
- Buildings
- Vegetation
- Sky
- Traffic Signs

This provides environmental context beyond object detection.

### Lane Detection
Lane markings are extracted using classical computer vision techniques (Canny Edge + Hough Transform) to estimate road structure and lane boundaries.

### Distance Estimation
Approximate distance to detected objects is estimated based on bounding box size.

### Collision Risk Detection
The system evaluates potential collision risks when objects appear too close to the ego vehicle.

Objects within critical range are labeled:

- WARNING
- COLLISION RISK

## Dataset

This project uses the **Cityscapes dataset**, a benchmark dataset for urban scene understanding.

Cityscapes provides high-resolution street scenes from multiple European cities and includes annotations for semantic segmentation.

Dataset link:

https://www.cityscapes-dataset.com/

## Example Output


The system produces visualizations combining:

- object bounding boxes
- semantic segmentation overlays
- lane detection lines
- distance labels
- collision warnings

## Example Results

### Scene 1
![Result 1]("C:\Users\Nimra Tariq\Downloads\Screenshot_1.png")

### Scene 2
![Result 2](results/result2.png)

### Scene 3
![Result 3](results/result3.png)

### Scene 4
![Result 4](results/result4.png)

### Scene 5
![Result 5](results/result5.png)


## Architecture

The perception pipeline follows this structure:

Camera Image  
↓  
Object Detection (YOLOv8)  
↓  
Semantic Segmentation (SegFormer)  
↓  
Lane Detection  
↓  
Distance Estimation  
↓  
Collision Risk Evaluation  
↓  
Scene Understanding

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/autonomous-vehicle-perception-stack.git
cd autonomous-vehicle-perception-stack
