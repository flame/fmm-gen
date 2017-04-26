#   Utility function: modified from https://github.com/arbenson/fast-matmul
#   Copyright (c) 2014-2015, Sandia Corporation
#   All rights reserved.
#
#   This file is part of fast-matmul and is under the BSD 2-Clause License, 
#   which can be found in http://opensource.org/licenses/BSD-2-Clause.

import sys

def is_one(x):
    return x == 1 or x == 1.0 or x == '1' or x == '1.0'

def is_negone(x):
    return x == -1 or x == -1.0 or x == '-1' or x == '-1.0'

def is_nonzero(x):
    return x != 0 and x != 0.0 and x != -0.0 and x != '0' and x != '0.0' and x != '-0.0'

def contain_nontrivial( coeffs ):
    for coeff in coeffs:
        if ( ( not is_one( coeff ) ) and ( not is_negone( coeff ) ) and ( is_nonzero( coeff ) ) ):
            return True
    return False

def write_line(myfile, num_indent, code):
    myfile.write(' ' * 4 * num_indent + code + '\n')

def write_break(myfile, num_breaks=1):
    myfile.write('\n' * num_breaks)

def data_access( mat_name, ind="" ):
    return '%s[ i + j * ld%s ]' % ( mat_name + str(ind), mat_name )

def transpose(coeffs):
    return [[x[i] for x in coeffs] for i in range(len(coeffs[0]))]

def parse_coeff(coeff):
    ''' Parse a coefficient. The grammar is:
        
        * --> *i | -* | *p | [a-z] | [floating point number]
        p --> 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
        *i --> 1 / (*)
        -* --> -(*)
        *p --> (*)^p

        So -x2i is parsed as - (1 / ((x)^2))
    '''
    coeff = coeff.strip()
    # First try to convert to float
    try:
        val = float(coeff)
        return coeff
    except:
        pass
    
    # Parameterized coefficient
    if len(coeff) == 1:
        # Coeff is like 'x'.  We will use 'x' instead of whatever is provided.
        # For now, this means that we only support one paramterized coefficient.
        return 'x'
    elif coeff[0] == '(':
        assert(coeff[-1] == ')')
        expr = coeff[1:-1].split('+')
        return '(' + ' + '.join([parse_coeff(e) for e in expr]) + ')'
    elif coeff[0] == '-':
        return '-(%s)' % parse_coeff(coeff[1:])
    elif coeff[-1] == 'i':
        return '1.0 / (%s)' % parse_coeff(coeff[:-1])
    else:
        # Test for a multiplier out in front
        try:
            mult = float(coeff[0])
            return '%s * (%s)' % (mult, parse_coeff(coeff[1:]))
        except:
            pass

        # Test for an exponent
        try:
            exp = int(coeff[-1])
            return ' * '.join([parse_coeff(coeff[:-1]) for i in xrange(exp)])
        except:
            raise Exception('Cannot parse coefficient: %s' % coeff)

def read_coeffs(filename):
    ''' Read the coefficient file.  There is one group of coefficients for each
    of the three matrices.

    filename is the name of the file from which coefficients are read
    '''
    coeffs = []
    with open(filename, 'r') as coeff_file:
        curr_group = []
        for line in coeff_file:
            if line[0] == '#':
                if len(curr_group) > 0:
                    coeffs.append(curr_group)
                    curr_group = []
            else:
                curr_group.append([parse_coeff(val) for val in line.split()])
    coeffs.append(curr_group)
    # There should be three sets of coefficients: one for each matrix.
    if (len(coeffs) < 3):
        raise Exception('Expected three sets of coefficients!')
    return coeffs

def num_nonzero(arr):
    ''' Returns number of non-zero entries in the array arr. '''
    return len(filter(is_nonzero, arr))

