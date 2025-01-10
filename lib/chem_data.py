import yaml
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

atomic_data = yaml.safe_load(open(os.path.join(current_directory, 'atomic_data.yaml')))
for n, d in atomic_data.items():
    exact_mass_data = d['data']
    sorted_data = sorted(exact_mass_data, key=lambda x: x["Composition"], reverse=True)

    atomic_data[n]['data'] = sorted_data

atomic_to_exact_mass = {d['isotope']:d['data'][0]['ExactMass'] for d in atomic_data.values()}

# def read_atomic_data():
#     atomic_data = yaml.safe_load(open(os.path.join(current_directory, 'atomic_data.yaml')))

#     for n, d in atomic_data.items():
#         exact_mass_data = d['data']
#         sorted_data = sorted(exact_mass_data, key=lambda x: x["Composition"], reverse=True)

#         atomic_data[n]['data'] = sorted_data

#     return atomic_data

def calc_exact_mass(elements):
    exact_mass = 0
    for el, num in elements.items():
        exact_mass += atomic_to_exact_mass[el] * num

    return exact_mass


def read_adduct_type_data():
    aduct_type_data = yaml.safe_load(open(os.path.join(current_directory, 'adduct_type_data.yaml')))

    return aduct_type_data

if __name__ == '__main__':
    atomic_data = read_atomic_data()
    print(atomic_data)