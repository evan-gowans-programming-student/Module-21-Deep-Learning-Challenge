# Module-21-Deep-Learning-Challenge

## **Analysis of Deep Learning Model for Predicting Donation Success**

---

## **Introduction**
The purpose of this analysis is to develop and optimize a deep learning model that predicts whether charitable donations will be successful based on organizational application data. Using TensorFlow and neural networks, the goal was to improve accuracy beyond 75 percent through multiple optimization techniques, including feature selection, scaling, model architecture improvements, and hyperparameter tuning.

---

## **Data Preprocessing**
### **Cleaning the Dataset**
- Removed irrelevant features (`EIN` and `NAME`).  
- Grouped rare categories in `APPLICATION_TYPE` and `CLASSIFICATION` into `"Other"`.  
- Applied one-hot encoding to categorical variables to convert them into numeric format.  

### **Feature Scaling**
- Standardized data using `StandardScaler()`, ensuring all features have equal weight in the model.  

### **Training and Testing Sets**
- Split dataset into training (80 percent) and testing (20 percent) sets.  

---

## **Neural Network Model Design**
### **Initial Model Setup**
- Input Layer with 100 neurons, `LeakyReLU` activation.  
- Hidden Layers:  
  - Layer 1: 100 neurons, `LeakyReLU`  
  - Layer 2: 50 neurons, `Tanh`  
  - Layer 3: 25 neurons, `ReLU`  
  - Dropout Layer: 0.2 to reduce overfitting  
- Output Layer: 1 neuron, `Sigmoid` activation for binary classification  

### **Optimization Techniques**
- Increased neurons to improve learning.  
- Adjusted activation functions to enhance non-linearity.  
- Changed optimizer to Adam, improving gradient updates.  
- Tweaked dropout rate to 0.05 for better generalization.  
- Adjusted batch size and epochs (400 epochs, batch size 16).  

---

## **Results and Findings**
### **Evaluating Performance**
**Final Accuracy Achieved:** 72.69 percent  
- The optimized model showed improvement from earlier versions, which had around 72.49 percent accuracy.  
- Training accuracy plateaued due to early stopping, preventing overfitting.  

### **Answering Key Questions**
1. **What are the overall results of the model?**  
   - The final accuracy was 72.69 percent, improved from earlier iterations of around 71 percent.  

2. **Did the model achieve the desired target of 75 percent accuracy?**  
   - No, but it came close at 72.69 percent. Additional refinements could further improve accuracy.  

3. **What steps were taken to optimize the model?**  
   - Increased layers and neurons, adjusted activation functions and fine-tuned batch size and epochs. All of this was done multiple time to strive for a stronger accuracy

4. **How does this model compare to previous iterations?**  
   - It performs better than the initial version (72.49 percent accuracy) through added layers and adjusted hyperparameters.  

5. **What are potential weaknesses of this model?**  
   - Accuracy plateaued due to early stopping—allowing more epochs could further improve results.  
   - Overfitting is still somewhat present, and dropout rate may need further adjustment.  

6. **How could a different model be used for this problem?**  
   - A Random Forest model could be used, providing better feature importance analysis.  
   - Gradient Boosting (XGBoost) might outperform deep learning in structured data scenarios.  

---

## **Conclusion**
In this analysis, a neural network model was developed to predict successful charitable applications. Through multiple optimization techniques, accuracy improved to 72.69 percent but did not exceed 75 percent. Further refinements, additional layers, and alternative models such as XGBoost could enhance performance. Despite the challenges, the model demonstrates strong predictive capabilities for real-world applications in charity fundraising.

---
