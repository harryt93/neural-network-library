# Import external libraries.
import numpy as np
import cPickle
import random
import gzip
import bokeh
from IPython.display import clear_output
		
class SigmoidActivation(object):

	@staticmethod
	def fn(x):
		return 1/(1+np.exp(-x))
		
	@staticmethod
	def prime(x):
		return x*(1-x)
		
	def identify(self):
		return "SIGMOID"
		
	
class ReluActivation(object):

	@staticmethod
	def fn(x):
		return x * (x > 0)
		
	@staticmethod
	def prime(x):
		return 1 * (x > 0)
	
	def identify(self):
		return "RELU"
		
		
class TanhActivation(object):

	@staticmethod
	def fn(x):
		return np.tanh(x)
		
	@staticmethod
	def prime(x):
		return 1-(x**2)
		
	def identify(self):
		return "TANH"
		

class ArctanActivation(object):

	@staticmethod
	def fn(x):
		return np.arctan(x)
		
	@staticmethod
	def prime(x):
		return np.cos(x)**2
	
	def identify(self):
		return "ARCTAN"
		
		
class QuadraticCost(object):

	@staticmethod
	def fn(a, y):
		return 0.5*np.linalg.norm(a-y)**2

	@staticmethod
	def delta(a, y, activation_function):
		return (y-a) * activation_function.prime(a)
		
	def identify(self):
		return "QUADRATIC"
		

class CrossEntropyCost(object):

	@staticmethod
	def fn(a, y):
		return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

	@staticmethod
	def delta(a, y, activation_function):
		return (y-a)
		
	def identify(self):
		return "CROSS-ENTROPY"
		
		
# Plotting Function
from bokeh.plotting import figure, output_notebook, show

def plot_losses(evaluation_frequency, training_loss, validation_loss=None, evaluation_method="loss"):
    
	# Calculate x's, which represent epochs.
	# This is possible because you know how often evaluations were made.
	x = range(1, len(training_loss)*evaluation_frequency, evaluation_frequency)
    
	# Define the y's
	y_train_loss = training_loss
	y_valid_loss = validation_loss
    
	# Tell bokeh to output to the notebook
	output_notebook()
    
	# Use the correct words, depending on whether the evaluation method is an accuracy or a loss.
	if evaluation_method == "loss":
		
		# Create the figure.
		fig = figure(tools="pan, wheel_zoom, reset, save", title="Losses", x_axis_label="Epochs", y_axis_label="Loss")
		
		# Render training loss.
		fig.line(x, y_train_loss, legend="Training Loss")
		fig.circle(x, y_train_loss)
		
		# Render the validation loss if required.
		if validation_loss is not None:
			fig.line(x, y_valid_loss, legend="Validation Accuracy", line_color="red")
			fig.circle(x, y_valid_loss, fill_color="red", line_color="red")
		
	else:	
		# Create the figure.
		fig = figure(tools="pan, wheel_zoom, reset, save", title="Accuracy", x_axis_label="Epochs", y_axis_label="Accuracy")
		
		# Render training loss.
		fig.line(x, y_train_loss, legend="Training Accuracy")
		fig.circle(x, y_train_loss)
		
		# Render the validation loss if required.
		if validation_loss is not None:
			fig.line(x, y_valid_loss, legend="Validation Accuracy", line_color="red")
			fig.circle(x, y_valid_loss, fill_color="red", line_color="red")
    
    # Move the legend to the top left.
	fig.legend.location = "top_left"

	
	# Show the figure
	show(fig)
	
# Neural Network Class
class network(object):
    
	def __init__(self, layers, cost_function="quadratic", activation_function="sigmoid", regularisation_coefficient=0.0):
	
		# Error Checking
		################
		
		# Check that cost function is valid.
		valid_cost_functions = ["quadratic", "cross-entropy"]
		if cost_function not in valid_cost_functions:
			print ("I did not recognise your cost function, did you make a typo?")
			print ("Make sure it's one of {}".format(valid_cost_functions))
			return
		
		
		# Check that activation function is valid.
		valid_activation_functions = ["sigmoid", "relu", "tanh", "arctan"]
		if activation_function not in valid_activation_functions:
			print ("I did not recognise your activation function, did you make a typo?")
			print ("Make sure it's one of {}".format(valid_activation_functions))
			return

		
		# Store some parameters for later.
		self.layers = layers
		self.num_layers = len(layers)
        
		# Random.random returns random numbers from 0 to 1, drawn from a uniform distribution.
		# Therefore weights contain values between -1 and 1.
		# This line extracts sets of weights into the weight list by cleverly zipping adjacent layers.
		# Each i looks like (5, 12), where 5 is the number of neurons in the previous layer,
		# and 12 is the number of neurons in the next layer.
		
		# The np.sqrt(x) is the default weight initialisation scheme, where they're initialised as 
		# gaussian random variables with mean 0 and standard deviation 1/sqrt(n).

		self.w = [np.random.randn(x, y)/np.sqrt(x) for x, y in zip(layers[:-1], layers[1:])]
        
		
		# Same logic for the weights, applies to the biases. This line creates a vector of biases, where
		# each bias is a row vector. Note that no bias vector is created for the inputs.
		
		self.b = [np.random.randn(1, y) for y in layers[1:]]
		
        
		# This line inserts a dummy element into the start of the list. The only purpose of this
		# is to bump all of the weight matrices up by one in the indexing, purely to for notational
		# convenience later on. Now, w[1] refers to weights belonging to layer 1. In my notation, 
		# there is no such thing as weights in layer 0, hence the dummy variable.
		# This also applies to the biases.
        
		self.w = [0] + self.w
		self.b = [0] + self.b
		
		
		# Create the cost function to be used in this model.
		if cost_function == "quadratic":
			self.cost = QuadraticCost()
		elif cost_function == "cross-entropy":
			self.cost = CrossEntropyCost()
			
        
		# Create the activation function to be used in this model.
		if activation_function == "sigmoid":
			self.activation = SigmoidActivation()
		elif activation_function == "relu":
			self.activation = ReluActivation()
		elif activation_function == "tanh":
			self.activation = TanhActivation()
		elif activation_function == "arctan":
			self.activation = ArctanActivation()
			
			
		# Store the regularisation parameter. If the user doesn't set anything, then it defaults to 0, which is no regularisation.
		self.lmbda = regularisation_coefficient
			
			
		# Create some global parameters used elsewhere in the model.
		self.train_losses = []
		self.valid_losses = []
        
        
	def reset(self):
        
		# Reset all weights.
		self.w = [np.random.randn(x, y)/np.sqrt(x) for x, y in zip(self.layers[:-1], self.layers[1:])]
		self.w = [0] + self.w
		
		# Reset all biases.
		self.b = [np.random.randn(1, y) for y in self.layers[1:]]
		self.b = [0] + self.b
        
		# Reset losses
		self.train_losses = []
		self.valid_losses = []
	
	# This function is callable from the API, it runs the evaluation once given a test set and a method of evaluation. 
	# If loss is selected, it runs the evaluation with whatever loss function was chosen at model creation.
	def evaluate(self, test_set, evaluation_method="loss", dropout=None):
		
		if evaluation_method == "loss":
			return self.evaluate_loss(test_set, dropout)
		else:
			return self.accuracy(test_set, dropout)
        
        
	def evaluate_loss(self, test_set, dropout=None):
	
		# Unpack the tuple.
		x = test_set[0]
		y = test_set[1]
        
		# Run the forward pass
		predictions = self.forward_pass(x, dropout)
        
		# Calculate loss (with mean squared error in this case).
		loss = self.cost.fn(predictions, y)

        
		return loss
		
		
		
	def accuracy(self, test_set, dropout=None):
	
		# Unpack the tuple.
		x = test_set[0]
		y = test_set[1]
		
		# Run the forward pass
		predictions = self.forward_pass(x, dropout)
		
		# Calculate the accuracy
		score = 0
		for i in range(len(predictions)):
		
			index = np.argmax(predictions[i])
			if y[i][index] == 1:
				score += 1
				
		# Calculate score as a percentage.
		return float(score)/float(len(predictions))		
	
    
	def train(self, train_set, epochs, learning_rate, batch_size=32, progress=None, evaluation_method="loss", evaluation_frequency=100, validation_set=None, dropout = None, search=False):
        
		# Error Checking
		################
        
		# Check that the number of batches is correct.
		if (len(train_set[0]) % batch_size) is not 0:
			print ("Length of training set [{}] is not perfectly divisible by batch size [{}].".format(len(train_set[0]), batch_size))
			return
        
		# Check the labels are consistent with the features.
        
        
		# Housekeeping
		##############
        
		# Unpack train data
		x = train_set[0]
		y = train_set[1]
        
		# Unpack validation data, if possible.
		if validation_set is not None:
			x_valid = validation_set[0]
			y_valid = validation_set[1]
            
		# Extract data into batches
		x_batches = [x[k:k+batch_size] for k in xrange(0, len(x), batch_size)]
		y_batches = [y[k:k+batch_size] for k in xrange(0, len(x), batch_size)]
		
		# Extract an amount of data from the train set for evaluation, the same size as the validation set if provided, otherwise make it 1000.
		if validation_set is not None:
			x_evaluation_batch = x[:len(validation_set[0])]
			y_evaluation_batch = y[:len(validation_set[0])]
		else:
			x_evaluation_batch = x[:1000]
			y_evaluation_batch = y[:1000]
        
			
		# Evaluate losses at epoch 0
		############################
		
		# When plotting the results, the plot should start at epoch 0. This piece of code checks 
		# to see if there are any entries in the train_losses list, if not then it must be the first
		# run so it evaluates the losses and stores them. The plotter will later read these as epoch 0.
		# It uses the first batch of the training set, for training loss, and full test set for validation.
		
		if self.train_losses == []:
			self.train_losses.append(self.evaluate((x_evaluation_batch, y_evaluation_batch), evaluation_method, dropout)) 
			
			if validation_set is not None:
				self.valid_losses.append(self.evaluate((x_valid, y_valid), evaluation_method, dropout))
				
        
		# Train the network
		###################
		
		# Run for number of epochs.
		for i in range(1, epochs+1):
		
		
			# If progress mode is "graph" or "percentage_only", then this bit of code prints out the progress percentage, otherwise
			# the program hangs until training is finished, which can be quite frustrating.
			
			if (progress == "graph") or (progress =="percentage_only"):
				
				# Calculate progress through epochs as a percentage.
				progress_percentage = (float(i)/float(epochs))*100
				
				# Clear the output of the jupyter notebook.
				clear_output(wait=True)
				
				# Print the progress.
				print ("Training progress: {:.2f}%.".format(progress_percentage))

            
			# Run through all batches
			for x_batch, y_batch in zip(x_batches, y_batches):
            
				# Update the weights for this epoch.
				self.weight_update_from_batch(x_batch, y_batch, learning_rate, dropout)
            
			# Evaluate at regular intervals.
			if (i%evaluation_frequency) == 0:
                
				# Evaluate training loss using the first batch of the training set.
				training_loss = self.evaluate((x_evaluation_batch, y_evaluation_batch), evaluation_method, dropout)
				
				# Append to historical losses.
				self.train_losses.append(training_loss)	
				
				# Set validation loss to "" initially, so it can be used in the progress message.
				validation_loss = None
                
				# If required, evaluate validation loss
				if validation_set is not None:
					validation_loss = self.evaluate((x_valid, y_valid), evaluation_method, dropout)
					self.valid_losses.append(validation_loss)
					
				# If progress is set to "text", display progress, regardless of whether validation is running, it will take care of itself.
				if progress == "text":
					if evaluation_method == "loss":
						print ("Training/Validation Loss: {}/{}. ".format(training_loss, validation_loss))
					else:
						print ("Training/Validation Accuracy: {}/{}. ".format(training_loss, validation_loss))
        
		# Show results if progress is set to graph.
		###########################################
        
		# Clear the output if a graph is being displayed, otherwise don't because the textual output should remain.
		if evaluation_method == "graph":
			clear_output()
		
		if progress == "graph":
			# Plot the losses on a graph.
			if validation_set is None:
				plot_losses(evaluation_frequency, self.train_losses, evaluation_method)
			else:
				plot_losses(evaluation_frequency, self.train_losses, self.valid_losses, evaluation_method)
		
		if search is True:
			return (self.valid_losses)

    
	def show_settings(self):
		print ("#######################")
		print ("Model Architecture: {}".format(self.layers))
		print ("Cost Function: {}".format(self.cost.identify()))
		print ("Activation Function: {}".format(self.activation.identify()))
		print ("L2 Regularisation Lambda: {}".format(self.lmbda))
		
	
	def weight_update_from_batch(self, x, y, lr, dropout=None):

		# Calculate a delta of weights for the batch.
		delta_b, delta_w = self.backpropagate(x, y, dropout)
		
		# Update network weights
		self.w = [(1-lr*self.lmbda)*w +(float(lr)/len(x))*nw for w, nw in zip(self.w, delta_w)]
		
		# Update network biases.
		self.b = [b +(float(lr)/len(x))*nb for b, nb in zip(self.b, delta_b)]
        
	def forward_pass(self, x, dropout=None):
        
		# Legacy code.
		debug = False
		
		# Declare the vector for holding back prop values.
		a = [0]*self.num_layers
		z = [0]*self.num_layers
		
		if debug:
			print ("Initial 'a': {}".format(a)) 
			print ("Initial 'z': {}".format(z)) 
        
		# Forward Propagation
		#####################

		# This section sets the activations of the first layer, which are the inputs. 
		# It then iterates through each of the weights in the weight list, and calculates
		# the new activations for that layer. The final output is the activations from the
		# final layer.
            
		# Set the layer 0 activations to the inputs.
		a[0] = x
		
		if debug:
			print ("After assigning input vector 'a': {}".format(a)) 
            
		# Calculate activations for layers 1 onwards.
		for i in range(1, self.num_layers):
			z[i] = np.dot(a[i-1], self.w[i]) + self.b[i]
			a[i] = self.activation.fn(z[i])
			
			# Apply antidrop out.
			if dropout is not None:
				a[i] = self.antidropout(a[i], dropout)
			
			if debug:
				print ("At row {} 'a': {}".format(i, a)) 
				print ("At row {} 'z': {}".format(i, z)) 
            
		# Return the final layer activations.
		return a[-1]
    
	def dropout(self, a, dropout):
		dropped = a
		for i in range(len(dropped)):
			for j in range(len(dropped[i])):
				dropped[i][j] = dropped[i][j] if (random.uniform(0, 1) <= float(dropout)) else 0
		return dropped
		
	def antidropout(self, a, dropout):
		dropped = a
		for i in range(len(dropped)):
			dropped[i] = dropped[i] * (1.0/float(dropout))
		return dropped

	
	def backpropagate(self, x, y, dropout=None):
            
		# Declare the vector for holding back prop values.
		a = [0]*self.num_layers
		d = [0]*self.num_layers
		e = [0]*self.num_layers
		z = [0]*self.num_layers
        
		# Declare the weight list for populating. Add the dummy.
		w = [np.zeros(_w.shape) for _w in self.w[1:]]
		w = [0] + w
		
		# Declare the bias list for populating. Add the dummy.
		b = [np.zeros(_b.shape) for _b in self.b[1:]]
		b = [0] + b
    
		# Forward Propagation
		#####################

		# This section sets the activations of the first layer, which are the inputs. 
		# It then iterates through each of the weights in the weight list, and calculates
		# the new activations for that layer. The final output is the activations from the
		# final layer.
            
		# Set the layer 0 activations to the inputs.
		a[0] = x
            
		# Calculate activations for layers 1 onwards.
		for i in range(1, self.num_layers):
			z[i] = np.dot(a[i-1], self.w[i]) + self.b[i]
			a[i] = self.activation.fn(z[i])
			
			# Randomly dropout some of the neurons.
			if dropout is not None:
				a[i] = self.dropout(a[i], dropout)
                

		# Back Propagation
		##################
            
		# Back propagate error at the output to the delta in the final layer.
		d[-1] = self.cost.delta(a[-1], y, self.activation)
                
		# Loop from the penultimate layer back to layer 1, 
		# calculating errors and deltas as you go.            
		for i in range(self.num_layers-2, 0, -1):
                
			# Back propagate the delta from the layer in front, to the error in this layer.
			e[i] = np.dot(d[i+1], self.w[i+1].T)

			# Back propagate the error to the delta in this layer.
			d[i] = e[i] * self.activation.prime(a[i])

		# Store the new weight deltas
		for i in range(1, self.num_layers):  
			w[i] = np.dot(a[i-1].T, d[i])
			
		# Store the new bias deltas
		for i in range(1, self.num_layers): 

			# Collapse deltas.
			collapsed_d = d[i].mean(0)
			
			b[i] = collapsed_d
        
		# Return the weight deltas.
		return b, w
		
	def draw_network(self):
	
		# Import the libraries.
		from bokeh.plotting import figure, output_notebook, show
		import random

		# Output to the notebook.
		output_notebook()

		# Create the plot.
		fig = figure(tools="pan, wheel_zoom, reset, save")
		fig.axis.visible = False
		
		# Calculate neuron size.
		neuron_size = 500/max(self.layers)
		
		
		# Calculate Neuron Coordinates
		#################################
		
		# Create the neuron coordinate vectors.
		x = []
		y = []
		
		# Loop through all of the layers.
		for i in range(len(self.layers)):
			
			# Extend the x positions for that layer.
			x.extend([i]*self.layers[i])
			
			# Calculate the offset for the y positions if higher than layer 0.
			offset = 0.0
			if i > 0:
				offset = float(self.layers[0] - self.layers[i])/2			
			
			# Generate y positions
			y_extend = list(range(self.layers[i]))
			
			# Add the offset
			y_extend = [_y + offset for _y in y_extend]
			
			# Extend y positions.
			y.extend(y_extend)	

		
		# Calculate Weight Coordinates
		##############################
		
		# Create the weight line coordinate vectors.
		line_x = []
		line_y = []
		line_weight = []
		
		# Zip neuron coordinates into a list of neuron positions.
		neurons = zip(x, y)
		
		# Loop the layers (not including input layer)
		for i in range(1, len(self.layers)):
		
			# Calculate the index.
			index = 0
			for j in range(i):
				index += self.layers[j]
			
			size_current_layer = self.layers[i]
			size_previous_layer = self.layers[i-1]
				
			# Loop through neurons in layer 1.
			for neuron in neurons[index:index+size_current_layer]:
				
				# Loop through neurons in previous layer.
				for previous_neuron in neurons[index-size_previous_layer:index]:
				
					# Draw a line between them.
					line_x.append([previous_neuron[0], neuron[0]])
					line_y.append([previous_neuron[1], neuron[1]])
					line_weight.append(random.uniform(0.5, 1))
				
		
		# Draw the Network
		##################
		
		# Draw weights.
		fig.multi_line(line_x, line_y, color="#B4B4B4", alpha=0.5, line_width=2)
		
		# Draw neurons.
		fig.circle(x, y, size=neuron_size, color="#9facc4", line_color="#000000", line_width=2, alpha=1)	
		

		# show the results
		show(fig)

# This is a horribly horribly written function, please don't judge. Needs some major refactoring.
def search_parameter(train_set, valid_set, search_parameter, levels, epochs, evaluation_frequency, model_parameters, train_parameters):
	
	# Import the libraries.
	from bokeh.plotting import figure, output_notebook, show
	import random
	
	# Error Checking
	################
	
	# Check that the model_parameters dictionary contains layers, cost_function, activation_function, and regularisation_coefficient.
	try:
		test = model_parameters["layers"]
		test = model_parameters["cost_function"]
		test = model_parameters["activation_function"]
		test = model_parameters["regularisation_coefficient"]
	except:
		print ("I didn't get all the model parameters that I need, did you make a typo?")
		print ("You need to provide the following model parameters: {}".format(["layers", "cost_function", "activation_function", "regularisation_coefficient"]))
		return
		
	try:
		test = train_parameters["learning_rate"]
		test = train_parameters["batch_size"]
		test = train_parameters["dropout"]
	except:
		print ("I didn't get all the training parameters that I need, did you make a typo?")
		print ("You need to provide the following training parameters: {}".format(["learning_rate", "batch_size", "dropout"]))
		return
		
		
	# Check that the train_parameters dictionary contains epochs, learning_rate, batch_size, and evaluation_frequency.
	
	# Check that search parameter is right!
	valid_search_parameters = ["layers", "cost_function", "activation_function", "regularisation_coefficient", "learning_rate", "batch_size", "dropout"]
	if search_parameter not in valid_search_parameters:
		print ("I did not recognise your search parameter, did you make a typo?")
		print ("Make sure it's one of {}".format(valid_search_parameters))
		return
	
	
	# Housekeeping
	##############
	
	# Output to the notebook.
	output_notebook()

	# Create the plot.
	fig = figure(tools="pan, wheel_zoom, reset, save", title="Parameter Search for {}".format(search_parameter))
	
	
	# Create the y's for populating.
	ys = []
	
	# Create the legends for plotting.
	legends = []

	
	# Run the search
	################
	
	for level in levels:
	
		# Create a model based on the chosen model parameters.
		if search_parameter == "layers":
			net = network(layers=level, 
						cost_function=model_parameters["cost_function"], 
						activation_function=model_parameters["activation_function"],  
						regularisation_coefficient=model_parameters["regularisation_coefficient"])
		elif search_parameter == "cost_function":
			net = network(layers=model_parameters["layers"], 
						cost_function=level, 
						activation_function=model_parameters["activation_function"],  
						regularisation_coefficient=model_parameters["regularisation_coefficient"])
		elif search_parameter == "activation_function":
			net = network(layers=model_parameters["layers"], 
						cost_function=model_parameters["cost_function"], 
						activation_function=level,  
						regularisation_coefficient=model_parameters["regularisation_coefficient"])
		elif search_parameter == "regularisation_coefficient":
			net = network(layers=model_parameters["layers"], 
						cost_function=model_parameters["cost_function"], 
						activation_function=model_parameters["activation_function"],  
						regularisation_coefficient=level)
		else:
			net = network(layers=model_parameters["layers"], 
						cost_function=model_parameters["cost_function"], 
						activation_function=model_parameters["activation_function"],  
						regularisation_coefficient=model_parameters["regularisation_coefficient"])
			
		# Train the model based on the chosen model parameters.
		if search_parameter == "learning_rate":
			acc = net.train(train_set=train_set, epochs=epochs, learning_rate=level, 
							batch_size=train_parameters["batch_size"], progress="percentage_only", evaluation_method="accuracy",
							evaluation_frequency=evaluation_frequency, 
							validation_set=valid_set, dropout=train_parameters["dropout"], search=True)
		elif search_parameter == "batch_size":
			acc = net.train(train_set=train_set, epochs=epochs, learning_rate=train_parameters["learning_rate"], 
							batch_size=level, progress="percentage_only", evaluation_method="accuracy",
							evaluation_frequency=evaluation_frequency, 
							validation_set=valid_set, dropout=train_parameters["dropout"], search=True)
		elif search_parameter == "dropout":
			acc = net.train(train_set=train_set, epochs=epochs, learning_rate=train_parameters["learning_rate"], 
							batch_size=train_parameters["batch_size"], progress="percentage_only", evaluation_method="accuracy",
							evaluation_frequency=evaluation_frequency, 
							validation_set=valid_set, dropout=level, search=True)
		else:
			acc = net.train(train_set=train_set, epochs=epochs, learning_rate=train_parameters["learning_rate"], 
							batch_size=train_parameters["batch_size"], progress="percentage_only", evaluation_method="accuracy",
							evaluation_frequency=evaluation_frequency, 
							validation_set=valid_set, dropout=train_parameters["dropout"], search=True)
				 
		# Append the result to the y's.
		ys.append(acc)
		
		# Append the legend.
		legends.append("Level: {}".format(level))
	
	
	# Draw the graphs
	#################
	
	# Calculate the x axis based on evaluation parameters.
	x = range(1, len(ys[0])*evaluation_frequency, evaluation_frequency)
	
	# Draw lines
	for i in range(len(ys)):
		_color = (int(random.uniform(1, 255)), int(random.uniform(1, 255)), int(random.uniform(1, 255)))
		fig.line(x, ys[i], legend=legends[i], line_color=_color)
		
	# Move the legend to the bottom_right.
	fig.legend.location = "bottom_right"
	
	# Show the figure.
	show(fig)
		
# This function defines and returns some toy data for testing the network.
def toy_data_wiggle():

	train_x = np.array([[0.25, 0.5,  0.75, 0.3],
						[0.75, 0.25, 0.6,  0.5],
						[0.25, 0.8,  0.2,  0.25],
						[0.7,  0.75, 0.1,  0.4],
						[0.2,  0.5,  0.75, 0.25],
						[0.5,  0.8,  0.75, 0.25],
						[0.25, 0.75, 0.5,  0.5],
						[0.1,  0.5,  0.75, 0.9],
						[0.7,  0.75, 0.4,  0.5],
						[0.25, 0.5,  0.75, 0.2],
						[0.1,  0.75, 0.7,  0.8],
						[0.5,  0.4,  0.75, 0.25],
						[0.75, 0.5,  0.5,  0.9],
						[0.6,  0.75, 0.6,  0.5],
						[0.2,  0.7,  0.25, 0.75],
						[0.25, 0.3,  0.75, 0.5]])
						
	train_y = np.array([[0.6125, 1.0475, 0.5325, 0.915, 0.5525, 0.95, 0.7375, 1.01, 1.105, 0.5375, 1.015, 0.7125, 1.3125, 1.095, 0.6025, 0.6625]]).T
	
	test_x = np.array([[0.25, 0.2,  0.75, 0.4],
						[0.75, 0.25, 0.2,  0.5],
						[0.1, 0.75,  0.25,  0.8],
						[0.5,  0.75, 0.9,  0.25]])
						
	test_y = np.array([[0.5375, 0.8075, 0.61, 0.94]]).T
	
	# Package and return the data.
	return ((train_x, train_y), (None, None), (test_x, test_y))
	
	
def load_data():    
	f = gzip.open('data/mnist.pkl.gz', 'rb')
	training_data, validation_data, test_data = cPickle.load(f)
	f.close()
	return (training_data, validation_data, test_data)
	
# Horribly written, definitely needs refactoring at some point.
def toy_data_mnist():    
	mtrain_set, mvalidation_set, mtest_set = load_data()

	# Convert the data to the right shape.
	validation_features = mvalidation_set[0]
	validation_labels = mvalidation_set[1].reshape(10000, 1)
	
	new_validation_labels = []
	
	for label in validation_labels:
		vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		vec[label[0]] = 1
		new_validation_labels.append(vec)
		
	new_validation_labels = np.array(new_validation_labels)

	# Convert the data to the right shape.
	test_features = mtest_set[0]
	test_labels = mtest_set[1].reshape(10000, 1)
	
	new_test_labels = []
	
	for label in test_labels:
		vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		vec[label[0]] = 1
		new_test_labels.append(vec)

	new_test_labels = np.array(new_test_labels)

	# Convert the data to the right shape.
	features = mtrain_set[0]
	labels = mtrain_set[1].reshape(50000, 1)

	new_labels = []

	for label in labels:
		vec = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		vec[label[0]] = 1
		new_labels.append(vec)

	new_labels = np.array(new_labels)

	return [[features, new_labels], [validation_features, new_validation_labels], [test_features, new_test_labels]]