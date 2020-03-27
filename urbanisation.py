#!/usr/bin/env python3

import json
import sys
import time

import numpy as np

import scipy.spatial as spatial
import scipy.special as special
import scipy.stats as stats

SUSCEPTIBLE, EXPOSED, SYMPTOMATIC, ASYMPTOMATIC, REMOVED = 0, 1, 2, 3, 4
DAY, NIGHT = 0, 1
COMMUTER, NONCOMMUTER = 0, 1


def draw_population_sizes(rv):
    pop_size = np.exp(rv.rvs(GRID_SIZE**2))
    pop_size *= (POPULATION_SIZE + GRID_SIZE**2) / pop_size.sum()
    pop_size = pop_size.astype('int')
    
    if pop_size.sum() < POPULATION_SIZE:
        pop_size[np.random.choice(GRID_SIZE**2, POPULATION_SIZE-pop_size.sum(), replace=False)] += 1
    elif pop_size.sum() > POPULATION_SIZE:
        pop_size[np.random.choice(GRID_SIZE**2, pop_size.sum()-POPULATION_SIZE, replace=False)] -= 1

    return pop_size

def fit_population_size_distribution(data):
    return stats.gamma(*stats.gamma.fit(np.log(data)))

def matern_covariance(d, kappa):
    return 0.1 * (2**(kappa-1) * special.gamma(kappa))**(-1) * (d/5.0)**kappa * special.kv(kappa, d/5.0)

def setup_covariance_matrix(kappa):
    coords = np.transpose(np.divmod(np.arange(GRID_SIZE**2), GRID_SIZE))
    dist = spatial.distance.cdist(coords, coords)
    
    return np.where(dist>0.0, matern_covariance(dist, kappa), 0.1)

class Simulation:
    def __init__(self, block_pop_size):
        self.block_pop_size = block_pop_size

        self.population = np.zeros((GRID_SIZE**2, GRID_SIZE**2, 2, 5), 'int')
        self.time = 0
        self.step = 0.5
        self.tau = 1/10
        self.lambda_ = 1/1.9
        self.gamma = 1/3.0
        self.beta = 0.6

        self.p_EI = self.lambda_*self.step
        self.p_IIs = 0.67
        self.p_IsR = self.gamma*self.step
        self.p_IaR = self.gamma*self.step

        self.method = 'one'

        self.above_80_quantile = (lambda a: (a>np.quantile(a, 0.8)).nonzero()[0])(self.block_pop_size)
        self.largest_population = self.block_pop_size.argmax()

        coords = np.transpose(np.divmod(np.arange(GRID_SIZE**2), GRID_SIZE))
        dist = spatial.distance.cdist(coords, coords)
        N_j, N_i = np.meshgrid(block_pop_size, block_pop_size)
        
        n_commuters = np.where(dist>0.0, N_i**0.73 * N_j**0.51 / dist**1.22, 0)

        high, low = 1.0, 0.0
        while True:
            mid = (high+low)/2
            s = np.sum(np.floor(n_commuters*mid))
            if s > POPULATION_SIZE*0.1775:
                high = mid
            elif s < POPULATION_SIZE*0.1765:
                low = mid
            else:
                break

        n_commuters = (n_commuters*mid).astype('int')
        
        for home_location in range(GRID_SIZE**2):
            n_noncommuters = block_pop_size[home_location] - n_commuters[home_location].sum()
            assert n_noncommuters >= 0

            self.population[home_location, home_location, NONCOMMUTER, SUSCEPTIBLE] = n_noncommuters

            for work_location in n_commuters[home_location].nonzero()[0]:
                self.population[home_location, work_location, COMMUTER, SUSCEPTIBLE] = n_commuters[home_location, work_location]

        assert self.population.sum() == POPULATION_SIZE

    def do_step(self):
        if self.time % 2 == DAY:
            # seeding events

            arrival_locations = np.random.choice(self.above_80_quantile, 10, replace=False)
            self.population[arrival_locations, arrival_locations, COMMUTER, SYMPTOMATIC] += 1

            self.population[self.largest_population, self.largest_population, COMMUTER, SYMPTOMATIC] += 2

            # non-commuting travel
            
            for location in range(GRID_SIZE**2):
                destination_prob = 0.177/(1-0.177) * self.tau * self.block_pop_size / (POPULATION_SIZE-self.block_pop_size[location])
                destination_prob[location] = 1-(destination_prob.sum()-destination_prob[location])
                
                for state, n in enumerate(self.population[location,:,NONCOMMUTER].sum(axis=0)):
                    self.population[location,:,NONCOMMUTER,state] = np.random.multinomial(n, destination_prob)
        
        # infection dynamics
        
        for location in range(GRID_SIZE**2):
            if self.time % 2 == DAY:
                N = self.population[:,location,:]
                n_N = N.sum()
                n_Is = N[...,SYMPTOMATIC].sum()
                n_Ia = N[...,ASYMPTOMATIC].sum()
                p_SE = self.beta*self.step*(n_Is + 0.5*n_Ia) / n_N

                SE = np.random.binomial(N[...,SUSCEPTIBLE], p_SE)
                EI = np.random.binomial(N[...,EXPOSED], self.p_EI)
                EIs = np.random.binomial(EI, self.p_IIs)
                EIa = EI - EIs
                IsR = np.random.binomial(N[...,SYMPTOMATIC], self.p_IsR)
                IaR = np.random.binomial(N[...,ASYMPTOMATIC], self.p_IaR)
                self.population[:,location,:,SUSCEPTIBLE] += -SE
                self.population[:,location,:,EXPOSED] += SE-EI
                self.population[:,location,:,SYMPTOMATIC] += EIs-IsR
                self.population[:,location,:,ASYMPTOMATIC] += EIa-IaR
                self.population[:,location,:,REMOVED] += IsR+IaR
                        
            else:
                N_commuter = self.population[location,:,COMMUTER]
                N_noncommuter = self.population[:,location,NONCOMMUTER]
                
                n_N = N_commuter.sum() + N_noncommuter.sum()
                n_Is = N_commuter[...,SYMPTOMATIC].sum() + N_noncommuter[...,SYMPTOMATIC].sum()
                n_Ia = N_commuter[...,ASYMPTOMATIC].sum() + N_noncommuter[...,ASYMPTOMATIC].sum()
                p_SE = self.beta*self.step*(n_Is + 0.5*n_Ia) / n_N

                SE = np.random.binomial(N_commuter[...,SUSCEPTIBLE], p_SE)
                EI = np.random.binomial(N_commuter[...,EXPOSED], self.p_EI)
                EIs = np.random.binomial(EI, self.p_IIs)
                EIa = EI - EIs
                IsR = np.random.binomial(N_commuter[...,SYMPTOMATIC], self.p_IsR)
                IaR = np.random.binomial(N_commuter[...,ASYMPTOMATIC], self.p_IaR)
                self.population[location,:,COMMUTER,SUSCEPTIBLE] += -SE
                self.population[location,:,COMMUTER,EXPOSED] += SE-EI
                self.population[location,:,COMMUTER,SYMPTOMATIC] += EIs-IsR
                self.population[location,:,COMMUTER,ASYMPTOMATIC] += EIa-IaR
                self.population[location,:,COMMUTER,REMOVED] += IsR+IaR

                SE = np.random.binomial(N_noncommuter[...,SUSCEPTIBLE], p_SE)
                EI = np.random.binomial(N_noncommuter[...,EXPOSED], self.p_EI)
                EIs = np.random.binomial(EI, self.p_IIs)
                EIa = EI - EIs
                IsR = np.random.binomial(N_noncommuter[...,SYMPTOMATIC], self.p_IsR)
                IaR = np.random.binomial(N_noncommuter[...,ASYMPTOMATIC], self.p_IaR)
                self.population[:,location,NONCOMMUTER,SUSCEPTIBLE] += -SE
                self.population[:,location,NONCOMMUTER,EXPOSED] += SE-EI
                self.population[:,location,NONCOMMUTER,SYMPTOMATIC] += EIs-IsR
                self.population[:,location,NONCOMMUTER,ASYMPTOMATIC] += EIa-IaR
                self.population[:,location,NONCOMMUTER,REMOVED] += IsR+IaR

        self.time += 1

def main(kappa):
    with open('Folkemengde.json') as f:
        data = np.array(json.load(f)['dataset']['value'])
        
    pop_size_dist = stats.gamma(*stats.gamma.fit(np.log(data)))
    cov_matrix = setup_covariance_matrix(kappa)
    
    random_field = np.dot(np.linalg.cholesky(cov_matrix), stats.norm.rvs(size=GRID_SIZE**2))
    block_pop_size = draw_population_sizes(pop_size_dist)
    
    mapping = np.empty(GRID_SIZE**2, int)
    mapping[np.argsort(random_field)] = np.arange(GRID_SIZE**2)
    block_pop_size = np.take(np.sort(block_pop_size), mapping)

    simulation = Simulation(block_pop_size)

    while simulation.time < 730:
        print('[%s]' % time.ctime(), ['day', 'night'][simulation.time % 2], simulation.time // 2)
        print('S', simulation.population[...,SUSCEPTIBLE].sum(), ' / E', simulation.population[...,EXPOSED].sum(), ' / Is', simulation.population[...,SYMPTOMATIC].sum(), ' / Ia', simulation.population[...,ASYMPTOMATIC].sum(), ' / R', simulation.population[...,REMOVED].sum(), '\n')
        simulation.do_step()

if __name__ == '__main__':
    kappa, GRID_SIZE, POPULATION_SIZE = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    main(kappa=1.0)
