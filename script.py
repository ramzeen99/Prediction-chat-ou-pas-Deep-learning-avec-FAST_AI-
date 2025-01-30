from fastai.vision.all import *
import gradio as gr
path = untar_data(URLs.PETS)  # Télécharge et décompresse le dataset
files = get_image_files(path / "images")  # Récupère toutes les images
def is_cat(x): return x[0].isupper()  # Label : les noms d’images commençant par une majuscule sont des chats
dls = ImageDataLoaders.from_name_func(
    path, files, label_func=is_cat, item_tfms=Resize(224)
)
# Charger le modèle
learn_inf = load_learner('C:\\Users\\ramzeen\\Desktop\\DECTITION\\Pratical Deep learnong for corders FAST_AI----KAGGLE\\Pratical Deep Learning for corders Lesson_1__ FAST_AI\\pet_classifier.pkl')
# Fonction pour prédire
def classify_image(img):
    pred, _, probs = learn_inf.predict(img)
    return f"Prediction: {pred}, Probability: {probs.max():.4f}"
# Créer l'interface utilisateur
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),  # Remplacez gr.inputs.Image par gr.Image
    outputs="text"
)
# Lancer l'interface
interface.launch()