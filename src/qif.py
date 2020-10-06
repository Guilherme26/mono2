# coding: utf-8
import numpy as np
import pandas as pd


def prior_bayes_vulnerability(joint_dist):
        """
            Calculates the prior Bayes vulnerability given a joint distribution

            Parameters
            ----------
            joint_dist : pd.DataFrame
                Joint distribution table with variable X on the horizontal axis and
                variable Y on the vertical axis.

            Returns
            -------
            float
                Prior bayes vulnerability
        """
        
        return joint_dist.sum(axis=1).max()


def posterior_bayes_vulnerability(joint_dist, vulnerability_type='average'):
    """
        Calculates the posterior Bayes vulnerability given a joint distribution

        Parameters
        ----------
        joint_dist : pd.DataFrame
            Joint distribution table with variable X on the horizontal axis and
            variable Y on the vertical axis.

        vulnerability_type : str
            Select if the vulnerability to be calculated will be the maximum or average one.
            Usually the average vulnerabiltiy is more robust.
            (default is average)

        Returns
        -------
        posterior_vulnerability : float
            Posterior Bayes vulnerability
    """

    if vulnerability_type is 'average' or vulnerability_type is 'avg':
        Y = joint_dist.columns
        posterior_vulnerability = 0
        prior_vulnerability = joint_dist.sum(axis=1).max()
        for y in Y:
            posterior_vulnerability = posterior_vulnerability + joint_dist[y].max()
    elif vulnerability_type is 'maximum' or vulnerability_type is 'max':
        posterior_vulnerability = joint_dist.div(joint_dist.sum(axis=0), axis=1).values.max()
        prior_vulnerability = joint_dist.sum(axis=1).max()
    elif type(vulnerability_type) is str:
        raise ValueError("Vulnerabiltiy type not valid. Valid types are 'average' or 'maximum'")
    else:
        raise TypeError("Vulnerability type must be a string of values: 'maximum' or 'average'")
    
    return posterior_vulnerability


def calculate_leakage(posterior_vulnerability, prior_vulnerability, leakage_type='multiplicative'):
    """
        Compute the multiplicative or additive leakage

        Paramters
        ---------
        posterior_vulnerability : float
            Posterior vulnerability

        prior_vulnerability : float
            Prior vulnerability

        leakage_type : str
            The type of leakage to be calculated. The possible values are: 'multiplicative' or
            'additive' (default is multiplicative)

        Returns
        -------
        leakage : float
            The calculated leakage
    """

    if leakage_type is 'multiplicative' or leakage_type is 'mult':
        leakage = posterior_vulnerability/prior_vulnerability
    elif leakage_type is 'additive' or leakage_type is 'add':
        leakage = posterior_vulnerability - prior_vulnerability
    else:
        ValueError("Leakage type not valid. Valid types are 'multiplicative' or 'additive'")

    return leakage


def bayes_leakage(joint_dist, vulnerability_type='average', leakage_type='multiplicative'):
    """
        Computes the Bayes leakage

        Parameters
        ----------
        joint_dist : pd.DataFrame
            Joint distribution table with variable X on the horizontal axis and
            variable Y on the vertical axis.

        vulnerability_type : str
            Select if the vulnerability to be calculated will be the maximum or average one.
            Usually the average vulnerabiltiy is more robust.
            (default is average)

        leakage_type : str
            The type of leakage to be calculated. The possible values are: 'multiplicative' or
            'additive' (default is multiplicative)

        Returns
        -------
        leakage : float
            The computed Bayes leakage
    """

    prior_vulnerability = prior_bayes_vulnerability(joint_dist)
    posterior_vulnerability = posterior_bayes_vulnerability(joint_dist, vulnerability_type)
    leakage = calculate_leakage(posterior_vulnerability, prior_vulnerability, leakage_type)
    return leakage


class BayesLeakage(object):
    """
        Calculates the Bayes leakage of a joint distribution.
        So far it only considers the Bayes leakage whose gain function is the
        gain identity function.

        Attributes
        ----------
        data : pandas.DataFrame
            Data table

        Methods
        -------
        joint_distribution(x, y)
            Calculates the joint distribution of variables X and Y

        compute_flows(x, y='target', flow='multiplicative', verbose=False)
            Computes the direct and reverse flows
    """
    
    def __init__(self, data):
        self.data = data


    def joint_distribution(self, x, y):
        """
            Calculates the joint distribution of variables X and Y.

            Parameters
            ----------
            x : str
                Name of the independent variable X

            y : str
                Name of the dependent variable Y

            Returns
            -------
            joint_dist : pandas.DataFrame
                Joint distribution table with variable X on the horizontal axis and
                variable Y on the vertical axis.
        """

        joint_dist = self.data.groupby([x, y]).size().unstack()
        joint_dist.columns = joint_dist.columns.get_level_values(0)
        joint_dist = joint_dist.div(joint_dist.sum().sum())
        joint_dist.fillna(0, inplace=True)
        return joint_dist

    
    def compute_flows(self, x, y='target', vulnerability_type='average', leakage_type='multiplicative', verbose=False):
        """
            Computes the direct and reverse flows

            Parameters
            ----------
            x : str
                Name of the independent variable X

            y : str
                Name of the dependent variable Y

            flow : str
                The type of flow: multiplicative or aditive (default is multiplicative)

            verbose : bool, optional
                Prints the flows

            Returns
            -------
            direct_flow : float
                Direct flow

            reverse_flow : float
                Reverse flow
        """
        joint_dist = self.joint_distribution(x, y)
        direct_flow = bayes_leakage(joint_dist, vulnerability_type=vulnerability_type, leakage_type=leakage_type)
        reverse_flow = bayes_leakage(joint_dist.T, vulnerability_type=vulnerability_type, leakage_type=leakage_type)

        if(verbose):
            print("Direct Flow: ", np.round(direct_flow, 2))
            print("Reverse Flow: ", np.round(reverse_flow, 2))
        
        return direct_flow, reverse_flow