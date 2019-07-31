sigmoid = x => 1 / (1 + Math.exp(-x))
sigmoid_derivative = x => x * (1 - x)

class Test
{
	constructor(_inputs, _expected_output)
	{
		this.inputs = _inputs;
		this.expected_output = _expected_output;

		return this;
	}
}

class Neuron
{
	constructor(_inputs)
	{
		this.synapses = [];

		_inputs.forEach(
			(input, index) =>
			{
				this.synapses.push(
					new Synapse(index)
				);
			}
		);

		return this;
	}

	guess(_inputs)
	{
		let guess = 0;

		_inputs.forEach(
			(input, index) =>
			{
				guess += input * this.synapses[index].weight;
			}
		);

		guess = sigmoid(guess);

		return guess;
	}

	train(_tests)
	{
		let guesses = [];

		_tests.forEach(
			(test, index) =>
			{
				let inputs = test.inputs;
				let expected_output = test.expected_output;

				let guess = this.guess(inputs);

				guesses.push(guess);
			}
		);

		this.synapses.forEach(
			(synapse, index) =>
			{
				synapse.adjust(_tests, guesses);
			}
		);
	}
}

class Synapse
{
	constructor(_input_index)
	{
		this.weight = 2 * Math.random() - 1;
		this.input_index = _input_index;

		return this;
	}

	adjust(_tests, _guesses)
	{
		let adjustments = [];

		_tests.forEach(
			(test, index) =>
			{
				let expected_output = test.expected_output;
				let guess = _guesses[index];
				let error = expected_output - guess;
				let adjustment = error * sigmoid_derivative(guess);

				adjustments.push(adjustment);
			}
		);

		let adjustment = 0;

		_tests.forEach(
			(test, index) =>
			{
				adjustment += test.inputs[this.input_index] * adjustments[index];
			}
		);
		
		this.weight += adjustment;
	}
}

let tests = [
	new Test([0, 0, 1], 0),
	new Test([1, 1, 1], 1),
	new Test([1, 0, 1], 1),
	new Test([0, 1, 1], 0)
];

let neuron = new Neuron(
	tests[0].inputs
);

for(let i = 0; i < 10000000; i++)
{
	neuron.train(tests);
}

tests.forEach(
	(test, index) => 
	{
		console.log(neuron.guess(test.inputs));
	}
);