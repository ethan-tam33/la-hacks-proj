## Marine Biology is Hard 🐠

The health of coral reefs is the major signifier of a marine ecosystem’s health. Unfortunately, assessing coral health is a tedious — and potentially dangerous task: it costs thousands of dollars in diving gear, expertise in underwater navigation, and considerable time investment. Moreover, it’s impossible to take the reef back to a lab for analysis. All knowledge about the reef is reserved for the videos or images that biologists like us take when we’re underwater.

## **Enter Reefer. 🪸**

Time underwater is precious. Instead of spending time taking images and analyzing health while underwater, we wanted to enable marine biologists to analyze while above ground. Reefer is a web app that takes in a single image of coral and outputs:

1. An upgraded super-resolution image for in-depth analysis via 2D NERFs
2. The health of the coral (whether it is bleached or not)
3. An interpretable analysis of why the model evaluated the way it did

Here is what we’re most proud of:

## Key Advances 🐡

- **Binary Classification of Coral:** as bleached or unbleached with **97% accuracy** -- beats current state-of-the-art (like those published in Nature!) binary classification models by 5%.
- **Gemini for interpretability and cross-validation:** Utilized Gemini's image processing abilities to explain key features that likely influenced the classifier's decision and to cross-check whether the classifier made the right decision
- **Super-Resolution:**

## How? 🐙

We built 4 key pieces. Here’s how:

1. The Classifier
    1. 
2. The Gemini Interpretability Interface: LLM’s are not just generators, they’re also interpreters
    1. Models are mysterious and hard to understand — we use Gemini to help interpret **why** our models classify the way they do, helping marine biologists understand what key features a previously black-boxed model is using to decide
    2. We pass in the image, classification, and confidence score; using a prompt-engineered message, we utilize the object-recognition features of Gemini to analyze what features led to the classification and possible reasons why the model is feeling unconfident. 
    3. **Gemini acts as a safeguard for red flags:** **If what Gemini finds in an image disagrees with what the classifier labels, it’ll tell you, and explain why! Here’s an engineered example (since our model luckily seldom misclassifies) we used by passing into our Gemini script an image of an unbleached coral (”colorful-coral-reef.jpg) and claiming it was bleached (0 = “bleached”); as you can see, Gemini correctly rebuts this classification and explains why!
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7fe118e1-6653-44b1-9629-3d76e4d029b0/c4626e39-eea3-4f9a-84e6-cf75d34bc7f5/Untitled.png)
        
3. Super-resolution via 2D NERFs:
    1. 
4. The Website
    1. Built with Reefer
    2. home page: prompts the user for 3 photos (enough to generate NERF)
    3. graphic ai generated in spirit of gemini and classifer + nerf
    4. leads to page with prediction scores, NURF generated from images as well as information on classification created by gemini
    5. The website is built with the Reflex framework. We found the framework’s syntax and documentation easy to follow and execute. 
    6. 

## Stumbling Blocks

## What we learned

## What's next for Reefer (🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏🙏)

We wanna go here (https://www.ioes.ucla.edu/marine/richard-b-gump-south-pacific-research-station/) this summer and test it out fr (low key anyone at LAHacks got any connections???)
