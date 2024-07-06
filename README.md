Submitted for LAHacks -- https://www.youtube.com/watch?v=tAPZwWck18s&ab_channel=ArjunBanerjee

## Marine Biology is Hard 🐠

The health of coral reefs is the major signifier of a marine ecosystem’s health. Unfortunately, assessing coral health is a tedious — and potentially dangerous task: it costs thousands of dollars in diving gear, expertise in underwater navigation, and considerable time investment. Moreover, it’s impossible to take the reef back to a lab for analysis. All knowledge about the reef is reserved for the videos or images that biologists like us take when we’re underwater.

## Enter Reefer. 🪸

Time underwater is precious. Instead of spending time taking images and analyzing health while underwater, we wanted to enable marine biologists to analyze while above ground. Reefer is a web app that takes in a single image of coral and outputs:

1. The health of the coral (whether it is bleached or not)
2. An interpretable analysis of why the model evaluated the way it did

Here is what we’re most proud of:

## Key Advances 🐡

- **Binary Classification of Coral:** as bleached or unbleached with **97% accuracy** -- beats current state-of-the-art (like those published in Nature!) binary classification models by up to **13%**
- **Gemini for interpretability and cross-validation:** Utilized Gemini's image processing abilities to explain key features that likely influenced the classifier's decision and to cross-check whether the classifier made the right decision
- **Trying (and failing)** to build a model that takes a single image and translates it to a 3D model mesh (NERF) and a model that amplifies an image resolution to make super resolution.

## How? 🐙

We built 3 key pieces. Here’s how:

#### The Classifier

- Built from a 900 image dataset of bleached vs unbleached coral
- Validation set accuracy of **97%**; this is 13% higher than [existing CNN coral classification architecture](https://ieeexplore.ieee.org/document/9731905) and around the same as a [state of the art Nature paper published last year](https://www.nature.com/articles/s41598-023-46971-7) **without using bag-of-hybrid techniques** (albeit without localizing).
    
    Here’s the architecture layout:
    
    **1. Basic Block (BasicBlock):**
    
    - Essential building block with two convolutional layers and batch normalization for stable training and gradient flow.
    - Shortcut connection aids in vanishing gradient mitigation, crucial for capturing complex features.
    
    **2. ResNet Model (ResNet):** 
    
    - Stacks Basic Blocks to learn hierarchical features from simple to complex, essential for accurate feature identification.
    - ResNet18 chosen for its balance between complexity and efficiency, configured with [2, 2, 2, 2] blocks per layer.
    
    **3. Forward Pass and Evaluation:**
    
    - Convolutional layers and ReLU activation transform input into meaningful features.
    - Trained using suitable loss function and optimizer, evaluated on metrics like accuracy and Cross Entropy Loss.
    1. **Inference:**
        - Probabilities derived from activations and vector to probability functions

#### The Gemini Interpretability Interface:

LLM’s are not just generators, they’re also interpreters

- Models are mysterious and hard to understand — we use Gemini to help interpret **why** our models classify the way they do, helping marine biologists understand what key features a previously black-boxed model is using to decide
- We pass in the image, classification, and confidence score; using a prompt-engineered message, we utilize the object-recognition features of Gemini to analyze what features led to the classification and possible reasons why the model is feeling unconfident.
- **Gemini acts as a safeguard for red flags:** If what Gemini finds in an image disagrees with what the classifier labels, it’ll tell you, and explain why! Here’s an engineered example (since our model luckily seldom misclassifies) we used by passing into our Gemini script an image of an unbleached coral (”colorful-coral-reef.jpg) and claiming it was bleached (0 = “bleached”); as you can see, Gemini correctly rebuts this classification and explains why!
    
    ![Untitled](https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/002/856/552/datas/gallery.jpg)
    

#### The Website:

- Built with **Reflex** framework
- Homepage: the interface allows users to drag an image of a coral reef to the page; three buttons on the home screen allow users to upload each image, clear uploaded images, and perform analysis. By clicking the “Analyze” button, users deploy the pretrained ML models we developed on the image they uploaded; they are then taken to a new page to display the results.

## We tried to build a lot of other features. We failed. And Learned.  🎣

**NERFS are hard.** We spent a large portion of our competition attempting to build and deploy NERF (Neual Radiance Field Model) **capable of developing a 3D mesh based on a single image**. We read through a lot of papers (***[Shape, Pose, and Appearance from a Single Image via Bootstrapped Radiance Field Inversion](https://arxiv.org/abs/2211.11674); [pixelNeRF: Neural Radiance Fields from One or Few Images](https://arxiv.org/abs/2012.02190)***) and implemented a few of them; unfortunately, we learned dependencies are quite difficult to sort out.

![Untitled](https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/002/857/029/datas/gallery.jpg) 

**2D Super-Resolution is also pretty hard** One unique feature of implicit neural representations (INRs) and NeRFs is the ability the represent discrete signals and volumes as a continuous function. Historically, neural networks have allowed for the approximation of functions. Using this property, we explored the possibility of using the **continuous representation** as a platform for an intelligent network which could **reduce blur and increase overall resolution**. At best, our methods based on papers (***[Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739),*** and ***[Instant Neural Graphics Primitives with a Multiresolution Hash Encoding](https://nvlabs.github.io/instant-ngp/))*** did not meet our standards of improvement and marginally improved resolution, so we chose not to proceed. However, with more time to experiment, we believe we could adjust hyper-parameters to achieve the desired effect! Below are a few demo images:

![Untitled](https://d112y698adiu2z.cloudfront.net/photos/production/software_photos/002/857/030/datas/gallery.jpg)

## What's next for Reefer 🙏

We’d like to continue implementing NERFs and 2D Super resolution!

We also wanna go here (https://www.ioes.ucla.edu/marine/richard-b-gump-south-pacific-research-station/) this summer and test it out fr (low key anyone at LAHacks got any connections???)
