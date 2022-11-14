#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tef

def cnn_steppable(input_shape, checkpoint_path):
    model = tef.keras.models.Sequential()

    # conv1
    model.add(tef.keras.layers.Conv2D(16, (5, 5), strides=(2, 2), input_shape=input_shape))
    model.add(tef.keras.layers.BatchNormalization())
    model.add(tef.keras.layers.ReLU())

    ## conv1
    #model.add(tef.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), input_shape=input_shape))
    #model.add(tef.keras.layers.BatchNormalization())
    #model.add(tef.keras.layers.ReLU())

    # conv2
    model.add(tef.keras.layers.Conv2D(16, (7, 7)))
    model.add(tef.keras.layers.BatchNormalization())
    model.add(tef.keras.layers.ReLU())

    # conv2
    model.add(tef.keras.layers.Conv2D(16, (7, 7)))
    model.add(tef.keras.layers.BatchNormalization())
    model.add(tef.keras.layers.ReLU())

    # conv3
    model.add(tef.keras.layers.Conv2D(16, (5, 5)))
    model.add(tef.keras.layers.BatchNormalization())
    model.add(tef.keras.layers.Dense(2, activation='softmax'))

    # 学習方法の設定
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    if checkpoint_path != "":
        model.load_weights(checkpoint_path+'/checkpoint_steppable_region')

    return model

def cnn_pose(input_shape, checkpoint_path):
    model = tef.keras.models.Sequential()

    # conv1
    model.add(tef.keras.layers.Conv2D(16, (5, 5), strides=2, input_shape=input_shape))
    model.add(tef.keras.layers.BatchNormalization())
    model.add(tef.keras.layers.ReLU())

    # conv2
    model.add(tef.keras.layers.Conv2D(16, (5, 5)))
    model.add(tef.keras.layers.BatchNormalization())
    model.add(tef.keras.layers.ReLU())

    # conv4
    model.add(tef.keras.layers.Conv2D(16, (5, 5)))
    model.add(tef.keras.layers.BatchNormalization())
    model.add(tef.keras.layers.Dense(2, activation='linear'))

    # 学習方法の設定
    model.compile(optimizer='adam',loss='mse',metrics=['mae', 'mse'])

    if checkpoint_path != "":
        model.load_weights(checkpoint_path+'/checkpoint_pose')

    return model

def cnn_height(input_shape, checkpoint_path):
    model = tef.keras.models.Sequential()

    # conv1
    model.add(tef.keras.layers.Conv2D(16, (5, 5), strides=2, input_shape=input_shape))
    model.add(tef.keras.layers.BatchNormalization())
    model.add(tef.keras.layers.ReLU())

    # conv2
    model.add(tef.keras.layers.Conv2D(16, (5, 5)))
    model.add(tef.keras.layers.BatchNormalization())
    model.add(tef.keras.layers.ReLU())

    # conv4
    model.add(tef.keras.layers.Conv2D(16, (5, 5)))
    model.add(tef.keras.layers.BatchNormalization())
    model.add(tef.keras.layers.Dense(1, activation='linear'))

    # 学習方法の設定
    model.compile(optimizer='adam',loss='mse',metrics=['mae', 'mse'])

    if checkpoint_path != "":
        model.load_weights(checkpoint_path+'/checkpoint_height')

    return model
