import numpy as np

def numpy_to_pmatrix(a, matrix_type='pmatrix', replace_jian=False, with_dollar=False):
    text = f'\begin{matrix_type}'
    text += '\n'
    for x in range(len(a)):
        for y in range(len(a[x])):
            num_str = str(a[x][y])
            if replace_jian == True and a[x][y] < 0:
                num_str = num_str.replace('-', '\jian ')
            text += num_str
            text += r' & '
        text = text[:-2]
        text += r'\\'
        text += '\n'
    text += f'\end{matrix_type}'
    if with_dollar == False:
        print(text)
    else:
        print("$"+text+"$")


def print_pauliSumOp(operator, round_level=0, num_per_line=3, replace_jian=False):
    idx = 0
    text = ""
    sign = ""
    for term in operator:  
        idx += 1
        coeff = term.coeffs[0]
        pauli = term.to_pauli_op().primitive.__str__()
        pauli = "{" + pauli + "}"
        if np.isclose(coeff.imag, 0):
            coeff = np.real(coeff)
        coeff = np.round(coeff, round_level)
        if round_level == 0:
            coeff = int(coeff)
        coeff_str = str(coeff)   
        
        if idx != 1:
            if coeff < 0:
                sign = "-" if replace_jian==False else "\jian \,"
                coeff_str = coeff_str[1:]
            else:
                sign = "+"
        token = "&" if (idx % num_per_line) == 1 else ""
        text += f"{sign}{token}{coeff_str}\, \\mathrm{pauli }\,  "
        if (idx % num_per_line) == 0:
            text += "\\nonumber"
            text += r'\\'
            text += '\n' 
        
    print(text)
