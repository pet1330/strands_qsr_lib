# -*- coding: utf-8 -*-

from qsrrep_hmms.hmm_abstractclass import HMMAbstractclass
import numpy as np

class petes_hmm(HMMAbstractclass):

    _state_list = [u'egg,tomato:n2&tomato,ham:s1&tomato,egg:s2&ham,tomato:n1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:n1&tomato,ham:s1&tomato,egg:s1&ham,tomato:n1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:n1&tomato,ham:sw0&tomato,egg:s1&ham,tomato:ne0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:n1&tomato,ham:sw1&tomato,egg:s1&ham,tomato:ne1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:ne1&tomato,ham:w0&tomato,egg:sw1&ham,tomato:e0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:ne1&tomato,ham:nw0&tomato,egg:sw1&ham,tomato:se0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:n1&tomato,ham:po&tomato,egg:s1&ham,tomato:po&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:n0&tomato,ham:nw0&tomato,egg:s0&ham,tomato:se0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:n0&tomato,ham:po&tomato,egg:s0&ham,tomato:po&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:n0&tomato,ham:n0&tomato,egg:s0&ham,tomato:s0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:po&tomato,ham:n0&tomato,egg:po&ham,tomato:s0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:nw0&tomato,ham:n0&tomato,egg:se0&ham,tomato:s0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:po&tomato,ham:n1&tomato,egg:po&ham,tomato:s1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:w0&tomato,ham:n1&tomato,egg:e0&ham,tomato:s1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:w0&tomato,ham:ne1&tomato,egg:e0&ham,tomato:sw1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:s0&tomato,ham:n1&tomato,egg:n0&ham,tomato:s1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:s1&tomato,ham:n1&tomato,egg:n1&ham,tomato:s1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:se0&tomato,ham:n1&tomato,egg:nw0&ham,tomato:s1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:e0&tomato,ham:nw1&tomato,egg:w0&ham,tomato:se1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:po&tomato,ham:nw1&tomato,egg:po&ham,tomato:se1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:po&tomato,ham:nw0&tomato,egg:po&ham,tomato:se0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:ne0&tomato,ham:nw0&tomato,egg:sw0&ham,tomato:se0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:nw1&tomato,ham:po&tomato,egg:se1&ham,tomato:po&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:nw1&tomato,ham:e0&tomato,egg:se1&ham,tomato:w0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:nw1&tomato,ham:se0&tomato,egg:se1&ham,tomato:nw0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:po&tomato,ham:se0&tomato,egg:po&ham,tomato:nw0&ham,egg:nw0&egg,ham:se0',
 u'egg,tomato:po&tomato,ham:se0&tomato,egg:po&ham,tomato:nw0&ham,egg:nw1&egg,ham:se1',
 u'egg,tomato:e0&tomato,ham:se1&tomato,egg:w0&ham,tomato:nw1&ham,egg:nw1&egg,ham:se1',
 u'egg,tomato:e1&tomato,ham:se0&tomato,egg:w1&ham,tomato:nw0&ham,egg:nw1&egg,ham:se1',
 u'egg,tomato:po&tomato,ham:s1&tomato,egg:po&ham,tomato:n1&ham,egg:n1&egg,ham:s1',
 u'egg,tomato:s0&tomato,ham:s1&tomato,egg:n0&ham,tomato:n1&ham,egg:n1&egg,ham:s1',
 u'egg,tomato:s1&tomato,ham:s1&tomato,egg:n1&ham,tomato:n1&ham,egg:n2&egg,ham:s2',
 u'egg,tomato:ne0&tomato,ham:nw1&tomato,egg:sw0&ham,tomato:se1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:nw0&tomato,ham:n1&tomato,egg:se0&ham,tomato:s1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:sw0&tomato,ham:n1&tomato,egg:ne0&ham,tomato:s1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:e0&tomato,ham:n1&tomato,egg:w0&ham,tomato:s1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:nw0&tomato,ham:ne0&tomato,egg:se0&ham,tomato:sw0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:nw1&tomato,ham:ne1&tomato,egg:se1&ham,tomato:sw1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:nw1&tomato,ham:ne0&tomato,egg:se1&ham,tomato:sw0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:nw1&tomato,ham:e1&tomato,egg:se1&ham,tomato:w1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:n1&tomato,ham:se0&tomato,egg:s1&ham,tomato:nw0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:n1&tomato,ham:s0&tomato,egg:s1&ham,tomato:n0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:n1&tomato,ham:w0&tomato,egg:s1&ham,tomato:e0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:po&tomato,ham:ne1&tomato,egg:po&ham,tomato:sw1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:po&tomato,ham:ne1&tomato,egg:po&ham,tomato:sw1&ham,egg:sw1&egg,ham:ne1',
 u'egg,tomato:po&tomato,ham:n1&tomato,egg:po&ham,tomato:s1&ham,egg:sw1&egg,ham:ne1',
 u'egg,tomato:se0&tomato,ham:nw1&tomato,egg:nw0&ham,tomato:se1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:se0&tomato,ham:nw0&tomato,egg:nw0&ham,tomato:se0&ham,egg:po&egg,ham:po',
 u'egg,tomato:e0&tomato,ham:w0&tomato,egg:w0&ham,tomato:e0&ham,egg:po&egg,ham:po',
 u'egg,tomato:po&tomato,ham:po&tomato,egg:po&ham,tomato:po&ham,egg:po&egg,ham:po',
 u'egg,tomato:po&tomato,ham:e0&tomato,egg:po&ham,tomato:w0&ham,egg:sw1&egg,ham:ne1',
 u'egg,tomato:po&tomato,ham:e0&tomato,egg:po&ham,tomato:w0&ham,egg:sw0&egg,ham:ne0',
 u'egg,tomato:po&tomato,ham:e0&tomato,egg:po&ham,tomato:w0&ham,egg:w0&egg,ham:e0',
 u'egg,tomato:po&tomato,ham:se0&tomato,egg:po&ham,tomato:nw0&ham,egg:w0&egg,ham:e0',
 u'egg,tomato:ne0&tomato,ham:se0&tomato,egg:sw0&ham,tomato:nw0&ham,egg:w0&egg,ham:e0',
 u'egg,tomato:ne1&tomato,ham:s0&tomato,egg:sw1&ham,tomato:n0&ham,egg:w1&egg,ham:e1',
 u'egg,tomato:ne1&tomato,ham:s1&tomato,egg:sw1&ham,tomato:n1&ham,egg:w1&egg,ham:e1',
 u'egg,tomato:e2&tomato,ham:s1&tomato,egg:w2&ham,tomato:n1&ham,egg:w1&egg,ham:e1',
 u'egg,tomato:nw0&tomato,ham:ne1&tomato,egg:se0&ham,tomato:sw1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:n1&tomato,ham:se1&tomato,egg:s1&ham,tomato:nw1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:s1&tomato,ham:n2&tomato,egg:n1&ham,tomato:s2&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:ne1&tomato,ham:nw1&tomato,egg:sw1&ham,tomato:se1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:se1&tomato,ham:n2&tomato,egg:nw1&ham,tomato:s2&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:n2&tomato,ham:sw1&tomato,egg:s2&ham,tomato:ne1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:po&tomato,ham:e0&tomato,egg:po&ham,tomato:w0&ham,egg:w1&egg,ham:e1',
 u'egg,tomato:e0&tomato,ham:se0&tomato,egg:w0&ham,tomato:nw0&ham,egg:w1&egg,ham:e1',
 u'egg,tomato:e1&tomato,ham:se0&tomato,egg:w1&ham,tomato:nw0&ham,egg:w1&egg,ham:e1',
 u'egg,tomato:po&tomato,ham:ne0&tomato,egg:po&ham,tomato:sw0&ham,egg:sw0&egg,ham:ne0',
 u'egg,tomato:po&tomato,ham:e1&tomato,egg:po&ham,tomato:w1&ham,egg:w0&egg,ham:e0',
 u'egg,tomato:e1&tomato,ham:s0&tomato,egg:w1&ham,tomato:n0&ham,egg:nw1&egg,ham:se1',
 u'egg,tomato:e1&tomato,ham:s1&tomato,egg:w1&ham,tomato:n1&ham,egg:nw1&egg,ham:se1',
 u'egg,tomato:ne0&tomato,ham:po&tomato,egg:sw0&ham,tomato:po&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:w1&tomato,ham:ne1&tomato,egg:e1&ham,tomato:sw1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:ne1&tomato,ham:w1&tomato,egg:sw1&ham,tomato:e1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:po&tomato,ham:ne0&tomato,egg:po&ham,tomato:sw0&ham,egg:w0&egg,ham:e0',
 u'egg,tomato:po&tomato,ham:e0&tomato,egg:po&ham,tomato:w0&ham,egg:nw0&egg,ham:se0',
 u'egg,tomato:po&tomato,ham:se1&tomato,egg:po&ham,tomato:nw1&ham,egg:nw1&egg,ham:se1',
 u'egg,tomato:se0&tomato,ham:s1&tomato,egg:nw0&ham,tomato:n1&ham,egg:n1&egg,ham:s1',
 u'egg,tomato:se1&tomato,ham:s1&tomato,egg:nw1&ham,tomato:n1&ham,egg:n1&egg,ham:s1',
 u'egg,tomato:e0&tomato,ham:se0&tomato,egg:w0&ham,tomato:nw0&ham,egg:nw1&egg,ham:se1',
 u'egg,tomato:n1&tomato,ham:nw0&tomato,egg:s1&ham,tomato:se0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:n0&tomato,ham:ne0&tomato,egg:s0&ham,tomato:sw0&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:sw0&tomato,ham:ne1&tomato,egg:ne0&ham,tomato:sw1&ham,egg:s1&egg,ham:n1',
 u'egg,tomato:se1&tomato,ham:n1&tomato,egg:nw1&ham,tomato:s1&ham,egg:s1&egg,ham:n1']


    def __init__(self):
        super(self.__class__, self).__init__()
        self.num_possible_states = len(self._state_list)

    def _qsr_to_symbol(self, qsr_data):
        """Transforms a list of qsr state chains to a list of lists of numbers according to the alphabet.
        Needs to be overridden by the specific QSR to handle the correct symbols.

        :return: List of lists containing the qsr input data as symbols from the alphabet
            E.g.: [[1,4,2,7],[0,5,3,8,5,1,3]]
        """
        if not type(qsr_data[0]) == list:
            return self._qsr_to_symbol([qsr_data])
        state_rep = []
        for element in qsr_data:
            state_rep.append([self._state_list.index(x) for x in element])
        return state_rep

    def _symbol_to_qsr(self, symbols):
        """Transforms a list of symbols to the corresponding qsr state chains.
        Needs to be overridden by the specific QSR to handle the correct symbols.

        :return: List of lists of qsr state chains
            E.g.: [['dc','po','o'],['dc','po']]
        """
        ret = []
        for s in symbols:
            ret.append([self._state_list[x] for x in s])
        return ret    

    # def _create_emission_matrix(self, size, **kwargs):
    #     """Creating an emission matrix with the highest prob along the diagonal
    #     and a "pseudo" prob for all other states.

    #     :param kwargs: empty

    #     :return: The emission probability matrix

    #     """
    #     emi = np.eye(size)
    #     # emi[emi == 0] = 0.0001

    #     return emi/emi.sum(axis=1).reshape(-1, 1)