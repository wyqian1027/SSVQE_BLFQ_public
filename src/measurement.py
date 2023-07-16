class PostMeasurement:
    
    def build(ansatz, pauli_str, reversed_ordering=True):
        if reversed_ordering:
            pauli_str = pauli_str[::-1].upper()
        else:
            pauli_str = pauli_str.upper()
        post_circ = ansatz.copy()
        for i in range(ansatz.num_qubits):
            if pauli_str[i] == 'X':
                post_circ.h(i)
            elif pauli_str[i] == 'Y':
                post_circ.sdg(i)
                post_circ.h(i)
        return post_circ


class MeasStr:
    
    def extract_state(state):
        assert type(state) == str
        if state[0] == '-':
            sign, state = -1, state[1:]
        elif state[0] == '+':
            sign, state = +1, state[1:]
        else:
            sign, state = +1, state
        return sign, state
    
    def __init__(self, state):
        self.sign, self.state = MeasStr.extract_state(state)
        assert type(self.state) == str and self.sign in [1, -1]
    
    def __radd__(self, other):
        return self

    def __add__(self, other):
        if other == 0:
            return self
        return MeasStrSum([self, other])
    
    def __mul__(self, other):
        new_state = self.state + other.state
        new_sign = self.sign*other.sign
        return MeasStr(new_state) if new_sign > 0 else MeasStr(f"-{new_state}")
    
    def __str__(self):
        return f'{"+" if self.sign == 1 else "-"}|{self.state}>'
    
    def __repr__(self):
        return self.__str__()
    
    def size(self):
        return len(self.state)
    
class MeasStrSum:
    
    def __init__(self, measStr_list):
        assert type(measStr_list) == list
        self.states = measStr_list
        self.size = self.states[0].size()
        
    def __add__(self, other):
        assert other.size == self.size
        return MeasStrSum(self.states + other.states)
    
    def __mul__(self, other):
        res = []
        for x in self.states:
            for y in other.states:
                res.append( x * y )
        return MeasStrSum(res)
    
    def __str__(self):
        return " ".join([x.__str__() for x in self.states])
    
    def __repr__(self):
        return self.__str__()
        
class PauliMeasStrSum:
    
    def mapper(pauliStr):
        pauliStr = pauliStr.upper()
        if pauliStr == 'I': return MeasStr('0') + MeasStr('1')
        if pauliStr in ['X', 'Y', 'Z']: return MeasStr('0') + MeasStr('-1')
        return PauliMeasStrSum.mapper(pauliStr[0]) * PauliMeasStrSum.mapper(pauliStr[1:])
        
    def __init__(self, pauliStr, coeff=1):
        self.pauliStr = pauliStr
        self.coeff = coeff
        self.measStr = PauliMeasStrSum.mapper(pauliStr)
        
    def __str__(self):
        return self.measStr.__str__()
    
    def __repr__(self):
        return self.__str__()
    
    def compute_with_count_dict(self, count_dict):
        shots = sum(count_dict.values())
        s = 0
        for el in self.measStr.states:
            if el.sign < 0:
                s -= count_dict.get(el.state, 0)
            else:
                s += count_dict.get(el.state, 0)
        s = s*self.coeff/shots
        return s