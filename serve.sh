#!/bin/bash

# Set environment variables for SageMaker
if [ "${SM_MODEL_DIR}" = "" ]; then
    export SM_MODEL_DIR=/opt/ml/model
fi

if [ "${SM_CHANNEL_TRAINING}" = "" ]; then
    export SM_CHANNEL_TRAINING=/opt/ml/input/data/training
fi

# Start the Gradio app
exec python3 app.py