string_input = (
    "f15.0;f24.0;f37.;f650; f4.034387870891734; f57.864648;842709; f240.442@457330"
)

# our tokenizer written in EarSharp has had a bit of a malfunction. we had expected a nicely formatted list of floats
# separated by commas, but got the above output. our song lyrics sentiment analysis model cannot process them as-is
# luckily we were able to confirm that the values themselves are correct in the dataset
# please create a script that takes the above input and breaks it into a numpy array of type double
# hint: every distinct value starts with the character f

#Mergin the two task in this part of the challenge

# well, it turns out that in order to be ingested by our model we have to solve a few more problems with our data
# the zeroeth column is the label, please split that out into it's own array
# the next two columns have their orders swapped, please put them in the correct spots
# our model expects an array of dimensionality (1,6) due to a legacy code issue,
# please make the vector conform to the input dimension

# please enter your solution here:
import numpy as np

string_input = "f15.0;f24.0;f37.;f650; f4.034387870891734; f57.864648;842709; f240.442@457330"


def solve_problem(raw_str: str):
        raw_str = raw_str.split('@')[0]  # get rid of everything after the '@' char
        raw_str = raw_str.strip()  # remove leading and trailing whitespaces if they exist (just in case)

        all_numbers = []
        for float_number_str in raw_str.split(
                ';'):  # split the string at ';' char to create an array we can iterate over

            if 'f' in float_number_str:  # there is a number that doesn't start with 'f' (842709), so we don't include it
                float_number_str = float_number_str.strip()  # get rid of whitespaces around
                float_number_str = float_number_str[1:]  # don't include the first char 'f'
                float_number = float(float_number_str)  # convert the float number string to a python float()
                all_numbers.append(float_number)

        # label is the first column
        label = all_numbers[0]
        # swap the first and second columns and append the rest of the numbers
        rest = all_numbers[2], all_numbers[1], *all_numbers[3:]

        return label, rest

# calling the function on input data
label, rest = solve_problem(string_input)
# validating the results
print('Label: ', label)
print('Data: ', np.array([rest]))
print('Data shape: ', np.array([rest]).shape)
