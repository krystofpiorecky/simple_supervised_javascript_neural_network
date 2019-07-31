sigmoid = x => 1 / (1 + Math.exp(-x))
sigmoid_derivative = x => x * (1 - x)

training_inputs = [
	[0,0,1],
	[1,1,1],
	[1,0,1],
	[0,1,1]
];

training_outputs = [
	0,
	1,
	1,
	0
];

synaptic_weights = [];

for(let i = 0; i < 3; i++)
	synaptic_weights.push(2 * Math.random() - 1);

let input_layer = training_inputs;

for(let k = 0; k < 10000000; k++)
{
	outputs = [];
	errors = [];
	adjustments = [];

	for(let i = 0; i < 4; i++)
	{
		value = 0;

		for(let j = 0; j < 3; j++)
			value += input_layer[i][j] * synaptic_weights[j];

		value = sigmoid(value);

		outputs.push(value);

		error = training_outputs[i] - value;
		errors.push(error);

		adjustment = error * sigmoid_derivative(value);
		adjustments.push(adjustment);
	}

	for(let i = 0; i < 3; i++)
	{
		value = 0;

		for(let j = 0; j < 4; j++)
			value += input_layer[j][i] * adjustments[j];

		synaptic_weights[i] += value;
	}
}

console.log(outputs);