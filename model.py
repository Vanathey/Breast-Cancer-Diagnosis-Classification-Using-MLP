{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "gJjBWjTw_0p3"
   },
   "source": [
    "Importing necessary libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "id": "8ztidrv5_8gm"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Dense\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "zI2eNRXF_-52"
   },
   "source": [
    "Load the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "id": "KCQezgXyADhW"
   },
   "outputs": [],
   "source": [
    "\n",
    "file_path = 'C:/Users/capta/Downloads/Model/wdbc.csv'\n",
    "\n",
    "data = pd.read_csv(file_path)\n",
    "columns = ['id', 'target'] + [f'feature_{i}' for i in range(1, 31)]\n",
    "data.columns = columns\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "# Encoding target variable ('M' -> 1, 'B' -> 0)\n",
    "label_encoder = LabelEncoder()\n",
    "data['target'] = label_encoder.fit_transform(data['target'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "L2kcCAxZAQCe"
   },
   "source": [
    "Identify features (X) and target variable (y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "id": "ACRXnyFYAMEf"
   },
   "outputs": [],
   "source": [
    "data.drop(columns=['id'], inplace=True)\n",
    "X = data.drop(columns=['target'])\n",
    "y = data['target']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "e2YDZ03bAN-u"
   },
   "source": [
    "Data preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "id": "U4k-EiffAZP2"
   },
   "outputs": [],
   "source": [
    "# Standardizing the features\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Tcv-VnNgAkm2"
   },
   "source": [
    "Data splitting (train:test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "id": "oTGlwhKeAqQV"
   },
   "outputs": [],
   "source": [
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "3zFzVAAbArzm"
   },
   "source": [
    "Create the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "id": "J8TmJtpRAtqQ"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\capta\\AppData\\Roaming\\Python\\Python312\\site-packages\\keras\\src\\layers\\core\\dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
      "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
     ]
    }
   ],
   "source": [
    "model = Sequential([\n",
    "    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),\n",
    "    Dense(1, activation='sigmoid')\n",
    "])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "W87tDxgcDByF"
   },
   "source": [
    "Setting hyperparameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "id": "4_pUTV3EDEEF"
   },
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "euvpD66LD9km"
   },
   "source": [
    "Train the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "id": "mI3n2xo-DLD9"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m2s\u001b[0m 22ms/step - accuracy: 0.6209 - loss: 1.0249 - val_accuracy: 0.6703 - val_loss: 0.7757\n",
      "Epoch 2/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.7068 - loss: 0.6830 - val_accuracy: 0.7912 - val_loss: 0.4914\n",
      "Epoch 3/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.8026 - loss: 0.4457 - val_accuracy: 0.8681 - val_loss: 0.3384\n",
      "Epoch 4/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step - accuracy: 0.8727 - loss: 0.3086 - val_accuracy: 0.9121 - val_loss: 0.2704\n",
      "Epoch 5/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9223 - loss: 0.2426 - val_accuracy: 0.9341 - val_loss: 0.2314\n",
      "Epoch 6/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 7ms/step - accuracy: 0.9401 - loss: 0.2086 - val_accuracy: 0.9560 - val_loss: 0.2088\n",
      "Epoch 7/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9284 - loss: 0.1957 - val_accuracy: 0.9560 - val_loss: 0.1922\n",
      "Epoch 8/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9461 - loss: 0.1718 - val_accuracy: 0.9560 - val_loss: 0.1782\n",
      "Epoch 9/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 8ms/step - accuracy: 0.9462 - loss: 0.1499 - val_accuracy: 0.9670 - val_loss: 0.1671\n",
      "Epoch 10/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9590 - loss: 0.1225 - val_accuracy: 0.9670 - val_loss: 0.1588\n",
      "Epoch 11/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9615 - loss: 0.1371 - val_accuracy: 0.9670 - val_loss: 0.1509\n",
      "Epoch 12/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9590 - loss: 0.1173 - val_accuracy: 0.9670 - val_loss: 0.1445\n",
      "Epoch 13/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9620 - loss: 0.1003 - val_accuracy: 0.9780 - val_loss: 0.1388\n",
      "Epoch 14/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9522 - loss: 0.1121 - val_accuracy: 0.9780 - val_loss: 0.1333\n",
      "Epoch 15/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9694 - loss: 0.0925 - val_accuracy: 0.9780 - val_loss: 0.1293\n",
      "Epoch 16/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9572 - loss: 0.1093 - val_accuracy: 0.9780 - val_loss: 0.1254\n",
      "Epoch 17/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9575 - loss: 0.1172 - val_accuracy: 0.9780 - val_loss: 0.1216\n",
      "Epoch 18/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9690 - loss: 0.1008 - val_accuracy: 0.9780 - val_loss: 0.1182\n",
      "Epoch 19/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9844 - loss: 0.0729 - val_accuracy: 0.9780 - val_loss: 0.1155\n",
      "Epoch 20/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9769 - loss: 0.0853 - val_accuracy: 0.9780 - val_loss: 0.1132\n",
      "Epoch 21/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9707 - loss: 0.0954 - val_accuracy: 0.9780 - val_loss: 0.1113\n",
      "Epoch 22/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9850 - loss: 0.0667 - val_accuracy: 0.9780 - val_loss: 0.1093\n",
      "Epoch 23/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9801 - loss: 0.0936 - val_accuracy: 0.9780 - val_loss: 0.1079\n",
      "Epoch 24/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9941 - loss: 0.0581 - val_accuracy: 0.9670 - val_loss: 0.1063\n",
      "Epoch 25/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9876 - loss: 0.0650 - val_accuracy: 0.9670 - val_loss: 0.1050\n",
      "Epoch 26/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9906 - loss: 0.0630 - val_accuracy: 0.9670 - val_loss: 0.1037\n",
      "Epoch 27/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9901 - loss: 0.0570 - val_accuracy: 0.9670 - val_loss: 0.1032\n",
      "Epoch 28/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 7ms/step - accuracy: 0.9718 - loss: 0.0967 - val_accuracy: 0.9670 - val_loss: 0.1019\n",
      "Epoch 29/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9858 - loss: 0.0529 - val_accuracy: 0.9670 - val_loss: 0.1005\n",
      "Epoch 30/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9896 - loss: 0.0556 - val_accuracy: 0.9780 - val_loss: 0.1006\n",
      "Epoch 31/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9887 - loss: 0.0603 - val_accuracy: 0.9780 - val_loss: 0.1001\n",
      "Epoch 32/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9876 - loss: 0.0547 - val_accuracy: 0.9780 - val_loss: 0.0993\n",
      "Epoch 33/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9907 - loss: 0.0509 - val_accuracy: 0.9780 - val_loss: 0.0990\n",
      "Epoch 34/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9846 - loss: 0.0676 - val_accuracy: 0.9780 - val_loss: 0.0983\n",
      "Epoch 35/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9885 - loss: 0.0490 - val_accuracy: 0.9670 - val_loss: 0.0980\n",
      "Epoch 36/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9934 - loss: 0.0511 - val_accuracy: 0.9670 - val_loss: 0.0980\n",
      "Epoch 37/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9947 - loss: 0.0460 - val_accuracy: 0.9670 - val_loss: 0.0981\n",
      "Epoch 38/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9953 - loss: 0.0485 - val_accuracy: 0.9670 - val_loss: 0.0979\n",
      "Epoch 39/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9903 - loss: 0.0613 - val_accuracy: 0.9670 - val_loss: 0.0983\n",
      "Epoch 40/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9888 - loss: 0.0589 - val_accuracy: 0.9670 - val_loss: 0.0983\n",
      "Epoch 41/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9938 - loss: 0.0444 - val_accuracy: 0.9670 - val_loss: 0.0976\n",
      "Epoch 42/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9902 - loss: 0.0468 - val_accuracy: 0.9670 - val_loss: 0.0977\n",
      "Epoch 43/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9989 - loss: 0.0355 - val_accuracy: 0.9670 - val_loss: 0.0971\n",
      "Epoch 44/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9973 - loss: 0.0291 - val_accuracy: 0.9670 - val_loss: 0.0978\n",
      "Epoch 45/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 7ms/step - accuracy: 0.9953 - loss: 0.0398 - val_accuracy: 0.9670 - val_loss: 0.0980\n",
      "Epoch 46/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9829 - loss: 0.0575 - val_accuracy: 0.9670 - val_loss: 0.0978\n",
      "Epoch 47/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9963 - loss: 0.0319 - val_accuracy: 0.9670 - val_loss: 0.0977\n",
      "Epoch 48/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 6ms/step - accuracy: 0.9982 - loss: 0.0301 - val_accuracy: 0.9670 - val_loss: 0.0977\n",
      "Epoch 49/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9891 - loss: 0.0501 - val_accuracy: 0.9670 - val_loss: 0.0991\n",
      "Epoch 50/50\n",
      "\u001b[1m23/23\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 5ms/step - accuracy: 0.9967 - loss: 0.0329 - val_accuracy: 0.9670 - val_loss: 0.0988\n"
     ]
    }
   ],
   "source": [
    "history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "vfj7bN0lDMVG"
   },
   "source": [
    "Evaluate the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "id": "4h4qeIqfDOOd"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m4/4\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 19ms/step\n"
     ]
    }
   ],
   "source": [
    "y_pred = (model.predict(X_test) > 0.5).astype(int)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "gw5_eOTrDWu1"
   },
   "source": [
    "Performance metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "id": "A9SimhmeDXeV"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification Report:\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.99      1.00      0.99        72\n",
      "           1       1.00      0.98      0.99        42\n",
      "\n",
      "    accuracy                           0.99       114\n",
      "   macro avg       0.99      0.99      0.99       114\n",
      "weighted avg       0.99      0.99      0.99       114\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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/owEEHBTQUFBkqRTp06V6PpffvlFXl5eql27tkN7ZGSkQkJC9Msvvzi016hRo0gfoaGhOn78+GVGXNSdd96pNm3a6P7771dERIT69u2rDz/88C+TwQtxxsXFFTlXv359HT16VKdPn3Zov/heQkNDJalU93LrrbcqMDBQH3zwgd577z21bNmyyHd5QWFhoaZMmaI6derIarWqcuXKqlKlirZv366TJ0+WeMyrrrqqVA98vPTSSwoLC9PWrVs1bdo0Va1atcSfBYA/IwEE3FRQUJCqVaumH374oVSfu/ghjEupUKFCse2GYVz2GBfWp13g5+entWvX6osvvtC9996r7du3684771Tnzp2LXPtP/JN7ucBqtap3796aO3euFi1adMnqnyS9+OKLSk5O1o033qh3331XK1as0MqVK3XNNdeUuNIp/fH9lMaWLVt0+PBhSdKOHTtK9VkA+DMSQMCNdevWTfv27dP69ev/9tro6GgVFhZqz549Du3Z2dk6ceKE/YneshAaGurwxOwFF1cZJcnLy0s33XSTJk+erF27dumFF17QqlWr9OWXXxbb94U4d+/eXeTcTz/9pMqVK8vf3/+f3cAl9OvXT1u2bNGpU6eKfXDmgo8++kgdOnTQ7Nmz1bdvX918883q1KlTke+kpMl4SZw+fVr9+/dXgwYN9OCDD2rixInatGlTmfUPwFxIAAE39sQTT8jf31/333+/srOzi5zft2+fXn75ZUl/TGFKKvKk7uTJkyVJXbt2LbO4atWqpZMnT2r79u32tkOHDmnRokUO1/3+++9FPnthQ+SLt6a5ICoqSk2bNtXcuXMdEqoffvhBn3/+uf0+naFDhw567rnn9OqrryoyMvKS11WoUKFIdXHBggX63//+59B2IVEtLlkurSeffFKZmZmaO3euJk+erJiYGCUmJl7yewSAv8JG0IAbq1WrlubNm6c777xT9evXd3gTyDfffKMFCxYoKSlJktSkSRMlJiZq1qxZOnHihNq1a6dvv/1Wc+fOVc+ePS+5xcjl6Nu3r5588kn16tVLQ4cO1ZkzZ/T666+rbt26Dg9BjB8/XmvXrlXXrl0VHR2tw4cPa/r06br66qvVtm3bS/b/3//+V126dFF8fLwGDhyos2fP6pVXXlFwcLDGjh1bZvdxMS8vL40ePfpvr+vWrZvGjx+v/v37q3Xr1tqx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",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1400x600 with 0 Axes>"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1400x600 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "print(\"Classification Report:\")\n",
    "print(classification_report(y_test, y_pred))\n",
    "\n",
    "# Plotting the confusion matrix\n",
    "conf_matrix = confusion_matrix(y_test, y_pred)\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "plt.title('Confusion Matrix')\n",
    "plt.show()\n",
    "\n",
    "# Plotting training and validation metrics\n",
    "plt.figure(figsize=(14, 6))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "Zg4wrnDvDev9"
   },
   "source": [
    "Loss and accuracy plots"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "6_ANUyJRDSVV"
   },
   "source": [
    "## Loss plot\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(history.history['loss'], label='Training Loss')\n",
    "plt.plot(history.history['val_loss'], label='Validation Loss')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Loss')\n",
    "plt.title('Loss vs. Epochs')\n",
    "plt.legend()\n",
    "\n",
    "# Accuracy plot\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(history.history['accuracy'], label='Training Accuracy')\n",
    "plt.plot(history.history['val_accuracy'], label='Validation Accuracy')\n",
    "plt.xlabel('Epochs')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.title('Accuracy vs. Epochs')\n",
    "plt.legend()\n",
    "\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
