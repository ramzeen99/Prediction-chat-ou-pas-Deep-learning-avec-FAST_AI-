from fastai.vision.all import *
path = untar_data(URLs.PETS)  # Télécharge et décompresse le dataset
files = get_image_files(path / "images")  # Récupère toutes les images
def is_cat(x): return x[0].isupper()  # Label : les noms d’images commençant par une majuscule sont des chats
dls = ImageDataLoaders.from_name_func(
    path, files, label_func=is_cat, item_tfms=Resize(224)
)
dls.show_batch()  # Affiche un lot d’images
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)  # Entraîne le modèle
'''img = PILImage.create(path / "images" / "Abyssinian_1.jpg")  # Exemple d’image
pred, _, probs = learn.predict(img)
print(f"Prediction: {pred}, Probability: {probs[1]:.4f}")'''