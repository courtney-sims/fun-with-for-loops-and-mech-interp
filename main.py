from transformer_lens import HookedTransformer
from transformer_lens.utils import to_numpy


def get_care_indices(neuron_activations, max_activation):
    care_indices = []
    for key, item in enumerate(neuron_activations):
        if item > max_activation:
            print(f'Activation greater than max at index {key}')
            care_indices.append(key)
    return care_indices


def extract_neuron_activation(model, layer, neuron, text):
    cache = {}

    def caching_hook(act, hook):
        cache["activation"] = act[0, :, neuron]

    model.run_with_hooks(
        text, fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", caching_hook)]
    )

    return to_numpy(cache["activation"])


def check_for_activation(model, layers, neurons, text, care_indices, max_activation):
    for layer in range(layers):
        for neuron in range(neurons):
            neuron_activations = extract_neuron_activation(
                model, layer, neuron, text)
            for index in care_indices:
                if neuron_activations[index] > max_activation:
                    print(f'Layer: {layer}, Neuron: {neuron}')


if __name__ == '__main__':
    # can have these defines passed as CLI args eventually
    origModel = HookedTransformer.from_pretrained('solu-1l')
    layer = 0
    neuron = 737
    text = ") a series of 2) approaches 3) are pursuing 4) ambition 5) fame 6) come to terms with 7) work out 8) singled out 9) personality 10) taken apart 11) at ease 12) observe 13) modest 14) application 15) curiosity"
    max_activation = 0.25

    testModel = HookedTransformer.from_pretrained('gpt2-small')
    layers = 1
    neurons = 3072

    neuron_activations = extract_neuron_activation(
        origModel, layer-1, neuron-1, text)
    print(f'Initial activations: {neuron_activations}')
    care_indices = get_care_indices(neuron_activations, max_activation)
    print(f'Care indices: {care_indices}')
    check_for_activation(testModel, layers, neurons, text,
                         care_indices, max_activation)
