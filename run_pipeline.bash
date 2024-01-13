#!/bin/bash

# Step 1: Delete the saved_graphs folder if it exists
if [ -d "saved_graphs" ]; then
    echo "Deleting the existing saved_graphs folder..."
    rm -rf saved_graphs
fi

# Step 2: Run the createdataset.py script
echo "Running createdataset.py to create and save the graph dataset..."
python createdataset.py

# Step 3: Run the train_model.py script
echo "Running train_model.py to train the model..."
python train_model.py

echo "Script execution completed."
