#!/usr/bin/env python

from abc import abstractmethod, ABCMeta
import numpy as np
import ghmm as gh
from copy import deepcopy


class HMMAbstractclass():
    """Abstract class for HMM generation"""
    __metaclass__ = ABCMeta

    def __init__(self):
        """Initialising the class.
        Setting self.num_possible_states. This has to be overridden by classes
        inheriting form this one.
        """

        # TODO: Use abstract property to make sure this is set.
        self.num_possible_states = None # This has to be set by the specific imprelementation

    def get_hmm(self, **kwargs):
        """Getter function to create and get the HMM.

        :param **kwargs:
            - qsr_seq: the sequence of QSRs. This should be a list of state chains, i.e. a list of lists

        :return: The trained HMM
        """

        # TODO: Some error checking

        # If no errors, create HMM
        return self._create(**kwargs)

    def get_samples(self, **kwargs):
        """Getter function to generate and get the amples from the HMM.

        :param kwargs:
            * max_length: The maximum length of the resulting samples. This will always be kept if possible.
            * num_samples: The number of samples to generate
            * hmm: The HMM from which to sample

        :return: A list of lists of samples
        """

        # TODO: Some error checking

        # If no errors, create samples
        return self._sample(**kwargs)

    def get_log_likelihood(self, **kwargs):
        """Getter function to compute and get the loglikelihood of the given
        samples to be generated by the iven HMM.

        :param kwargs:
            * qsr_seq: A list of lists of qsr sequences to check against the HMM
            * hmm: The to generate the loglikelihood for

        :return: The accumulated loglikelihood for all the given samples
        """

        # TODO: Some error checking

        # If no errors, calculate loglikelihood
        return self._log_likelihood(**kwargs)

    def get_num_possible_states(self):
        """Get the number of possiblr states

        :return: Number of possible states
        """
        return self.num_possible_states

    def _create_sequence_set(self, qsr_seq, symbols):
        """Creating a sequence set for training

        :param qsr_seq: the observation seqence of symbols according to the alphabet as a list of lists
        :param symbols: the alphabet of possible symbols

        :return: the sequence set for the given observations
        """
        return gh.SequenceSet(symbols, qsr_seq)

    def _create_transition_matrix(self, size, **kwargs):
        """Method for the creation of the transition probability matrix. Creates
        a uniformly distributed matrix. Please override if special behaviour
        is necessary.

        :return: uniform SIZExSIZE transition matrix as a numpy array
        """

        trans = np.ones([size,size])
        return trans/trans.sum(axis=1)

    def _create_emission_matrix(self, size, **kwargs):
        """Method for the creation of the emission probability matrix. Creates
        a uniformly distributed matrix. Please override if special behaviour
        is necessary.

        :return: uniform SIZExSIZE emission matrix as a numpy array
        """

        emi = np.ones([size,size])
        return emi/emi.sum(axis=1)

    @abstractmethod
    def _qsr_to_symbol(self, qsr_data):
        """Transforms a list of qsr state chains to a list of lists of numbers according to the alphabet.
        Needs to be overridden by the specific QSR to handle the correct symbols.

        :return: List of lists containing the qsr input data as symbols from the alphabet
            E.g.: [[1,4,2,7],[0,5,3,8,5,1,3]]
        """
        return

    @abstractmethod
    def _symbol_to_qsr(self, symbols):
        """Transforms a list of symbols to the corresponding qsr state chains.
        Needs to be overridden by the specific QSR to handle the correct symbols.

        :return: List of lists of qsr state chains
            E.g.: [['dc','po','o'],['dc','po']]
        """
        return

    def _generate_alphabet(self, num_symbols):
        """Generate a simple integer alphabet: [0:num_symbols-1]

        :param num_symbols: The number of different qsr symbols

        :return: The ghmm integer range object to be used as an alphabet
        """
        return gh.IntegerRange(0, num_symbols)

    def _train(self, seq, trans, emi, num_possible_states, pseudo_transitions=False, start_at_zero=False):
        """Uses the given parameters to train a multinominal HMM to represent
        the given seqences of observations. Uses Baum-Welch training.
        Please override if special training is necessary for your QSR.

        :param seq: the sequence of observations represented by alphabet symbols
        :param trans: the transition matrix as a numpy array
        :param emi: the emission matrix as a numpy array
        :param num_possible_states: the total number of possible states

        :return: the via baum-welch training generated hmm
        """

        print 'Generating HMM:'
        print seq
        print '\tCreating symbols...'
        symbols = self._generate_alphabet(num_possible_states)
        if start_at_zero:
            startprob = np.zeros(num_possible_states)
            startprob[0] = 1
        else:
            startprob = np.ones(num_possible_states)
            startprob = startprob/np.sum(startprob)
        print startprob
        print '\t\t', symbols
        print '\tCreating HMM...'
        hmm = gh.HMMFromMatrices(
            symbols,
            gh.DiscreteDistribution(symbols),
            trans.tolist(),
            emi.tolist(),
            startprob.tolist()
        )
        print '\tTraining...'
        hmm.baumWelch(self._create_sequence_set(seq, symbols))

        if pseudo_transitions:
            print '\tAdding pseudo transitions...'
            pseudo = deepcopy(trans)
            pseudo[pseudo > 0.] = 1.
            pseudo = pseudo/(float(len(seq)+1))

            trans_trained, emi, start = hmm.asMatrices()
            trans_trained = np.array(trans_trained)+pseudo

            hmm = gh.HMMFromMatrices(
                symbols,
                gh.DiscreteDistribution(symbols),
                trans_trained.tolist(),
                emi,
                start
            )

            hmm.normalize()

        return hmm


    def _create(self, **kwargs):
        """Creates and trains (using '_train') a HMM to represent the given qtc sequences.
        Main function to create and train the hmm. Please override with special
        behaviour if necessary.

        This function is called by the library to create the hmm.

        :param **kwargs:
            - qsr_seq: the sequence of QSRs. This should be a list of state chains, i.e. a list of lists

        :return: The trained HMM

        """

        state_seq = self._qsr_to_symbol(kwargs["qsr_seq"])
        trans = self._create_transition_matrix(size=self.get_num_possible_states(), **kwargs)
        emi = self._create_emission_matrix(size=self.get_num_possible_states(), **kwargs)
        hmm = self._train(
            state_seq,
            trans,
            emi,
            self.get_num_possible_states(),
            pseudo_transitions=kwargs["pseudo_transitions"],
            start_at_zero=kwargs["start_at_zero"]
        )
        print '...done'
        return hmm

    def _sample(self, **kwargs):
        """Gnerating samples from the trained HMM given a maximum sample length
        and thee total number of samples.

        :param kwargs:
            * max_length: The maximum length of the resulting samples. This will always be kept if possible.
            * num_samples: The number of samples to generate
            * hmm: The HMM from which to sample

        :return: A list of lists of samples

        """
        hmm = kwargs["hmm"]
        ret = []
        for i in range(int(kwargs["num_samples"])):
            ret.append(
                map(
                    self._generate_alphabet(num_symbols=self.num_possible_states).external,
                    hmm.sampleSingle(int(kwargs["max_length"]))
                )
            )
        return self._symbol_to_qsr(ret)


    def _log_likelihood(self, **kwargs):
        """Computeed the loglikelihood for the given sample(s) to be produced by
        the HMM.

        :param kwargs:
            * qsr_seq: A list of lists of qsr sequences to check against the HMM
            * hmm: The to generate the loglikelihood for

        :return: The accumulated loglikelihood for all the given samples
        """

        return kwargs["hmm"].loglikelihood(self._create_sequence_set(
            qsr_seq=self._qsr_to_symbol(kwargs["qsr_seq"]),
            symbols=self._generate_alphabet(num_symbols=self.num_possible_states)
        ))
