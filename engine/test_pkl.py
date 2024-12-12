import pickle

if __name__ == "__main__":
    with open('progs.pkl', 'rb') as file:
        loaded_list = pickle.load(file)

    for (vp, prog) in loaded_list:
        print("path:", vp)
        print(prog)

    print(len(loaded_list))