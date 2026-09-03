To Train the model, please run: python train.py
To test the model, please run: python test.py
To test the model please , change load model  function to load the pre-trained model weights.
for example :  model = joblib.load('model_name_e2phish_model.pkl')
	      selected_features = joblib.load('model_name_selected_features.pkl')	
                    