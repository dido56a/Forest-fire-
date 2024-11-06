import streamlit as st
import kagglehub
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.metrics import Recall, AUC
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Initialize Kaggle and login
st.title("Fire and Non-Fire Image Classification")

@st.cache_data
def download_datasets():
    kagglehub.login()
    phylake1337_fire_dataset_path = kagglehub.dataset_download('phylake1337/fire-dataset')
    aritrikghosh_forest_fire_path = kagglehub.dataset_download('aritrikghosh/forest-fire')
    return phylake1337_fire_dataset_path, aritrikghosh_forest_fire_path

st.write("Downloading datasets...")
phylake1337_fire_dataset_path, aritrikghosh_forest_fire_path = download_datasets()
st.write("Data source import complete.")

# Process images into DataFrame
@st.cache_data
def load_images():
    df = pd.DataFrame(columns=['path', 'label'])
    
    fire_images = []
    for dirname, _, filenames in os.walk('/kaggle/input/fire-dataset/fire_dataset/fire_images'):
        for filename in filenames:
            fire_images.append([os.path.join(dirname, filename), 'fire'])
    df = pd.concat([df, pd.DataFrame(fire_images, columns=['path', 'label'])], ignore_index=True)
    
    non_fire_images = []
    for dirname, _, filenames in os.walk('/kaggle/input/fire-dataset/fire_dataset/non_fire_images'):
        for filename in filenames:
            non_fire_images.append([os.path.join(dirname, filename), 'non_fire'])
    df = pd.concat([df, pd.DataFrame(non_fire_images, columns=['path', 'label'])], ignore_index=True)
    return df

st.write("Loading and processing images...")
df = load_images()
st.write("Images loaded successfully.")

# Display sample DataFrame
st.write("Sample Data:")
st.dataframe(df.head())

# Plot Distribution
st.subheader("Fire vs Non-Fire Image Distribution")
fig = px.scatter(data_frame=df, x=df.index, y='label', color='label', title='Distribution of fire and non-fire images along the length of the dataframe')
fig.update_traces(marker_size=2)
st.plotly_chart(fig)

label_counts = df['label'].value_counts()
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "xy"}, {"type": "pie"}]])
fig.add_trace(go.Bar(x=label_counts.index, y=label_counts.values, marker_color=['darkorange', 'green'], showlegend=False), row=1, col=1)
fig.add_trace(go.Pie(values=label_counts.values, labels=label_counts.index, marker=dict(colors=['darkorange', 'green'])), row=1, col=2)
st.plotly_chart(fig)

# Display sample images
def show_sample_images(label):
    st.subheader(f"Sample Images - {label.capitalize()}")
    data = df[df['label'] == label]
    pics = 6
    fig, ax = plt.subplots(pics // 2, 2, figsize=(15, 15))
    ax = ax.ravel()
    for i in range(pics):
        path = data.sample(1)['path'].values[0]
        img = image.load_img(path)
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

show_sample_images('fire')
show_sample_images('non_fire')

# Image Shape Analysis
df['height'], df['width'] = zip(*df.apply(lambda row: image.load_img(row['path']).size[::-1], axis=1))
st.write("Image Shape Analysis:")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [3, 0.5, 0.5]}, figsize=(15, 10))
sns.kdeplot(data=df[['height', 'width']], ax=ax1)
ax1.set_title('KDE Plot of Height and Width')
sns.boxplot(data=df, y='height', ax=ax2, color='skyblue')
ax2.set_title('Boxplot of Heights')
sns.boxplot(data=df, y='width', ax=ax3, color='orange')
ax3.set_title('Boxplot of Widths')
plt.tight_layout(rect=[0, 0, 1, 0.95])
st.pyplot(fig)

# Set up image generator
def create_image_generators(df):
    generator = ImageDataGenerator(rotation_range=20, width_shift_range=0.1, height_shift_range=0.1, shear_range=2, zoom_range=0.2, rescale=1/255, validation_split=0.2)
    image_size = (256, 256)
    train_gen = generator.flow_from_dataframe(df, x_col='path', y_col='label', target_size=image_size, class_mode='binary', subset='training')
    val_gen = generator.flow_from_dataframe(df, x_col='path', y_col='label', target_size=image_size, class_mode='binary', subset='validation')
    return train_gen, val_gen

train_gen, val_gen = create_image_generators(df)
st.write("Image generators created successfully.")

# Define CNN Model
def build_model():
    model = Sequential([
        Conv2D(32, (2, 2), activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (2, 2), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (2, 2), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Recall(), AUC()])
    return model

st.write("Building CNN model...")
model = build_model()
model.summary(print_fn=lambda x: st.text(x))

# Training the model
def train_model(model, train_gen, val_gen):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)
    history = model.fit(train_gen, epochs=15, validation_data=val_gen, callbacks=[early_stopping, reduce_lr_on_plateau])
    return history

if st.button("Train Model"):
    st.write("Training the model...")
    history = train_model(model, train_gen, val_gen)
    st.write("Training complete.")
    
    # Plot training metrics
    metrics_df = pd.DataFrame(history.history)
    fig = px.line(metrics_df, title="Training and Validation Metrics", labels={'index': 'Epoch'}, markers=True)
    fig.add_scatter(x=metrics_df.index, y=metrics_df['loss'], mode='lines+markers', name='loss')
    fig.add_scatter(x=metrics_df.index, y=metrics_df['accuracy'], mode='lines+markers', name='accuracy')
    st.plotly_chart(fig)
  
