
### Photo Cache - PennAppsXXIV ###

This is a web application called "Photo Cache" that allows users to upload a photo, and it calculates the similarity between the uploaded photo and a reference image. It utilizes a Siamese neural network model to perform image similarity comparison.

# Usage

1. Launch the app by running "python app.py" located in "/flaskapp" after installing prerequisites from the requirements.txt.
2. Visit the web application through your browser.
3. Try to find the actual location of the location given to you.
4. After finding the location, take a photo with similar angle and motive. Click the "Upload and Process" button.
4. The application will calculate the similarity between the uploaded photo and a reference image.
5. The result will be displayed on the webpage. If the similarity score is greater than some treshold, it will suggest retaking the photo from another angle or. Otherwise, it will display an interesting fact about said location.

# Technologies Used

Flask: This web application is built using the Flask framework.
HTML/CSS: The user interface of the web application is created using HTML and styled with CSS.
Siamese Neural Network: The application itself uses a pre-trained Siamese neural network to calculate image similarity.
Keras & tensorflow: Used for deep learning tasks, including loading and running the Siamese network.
Cloud ready: By packaging the project in conatiners using Docker the application is ready to deploy to a external webserver.

# Example
![explanation](https://github.com/svemyh/PennAppsXXIV/assets/40596752/164a0156-59ca-4e1d-a4fe-2e1a1d063cac)
