import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
from numpy.polynomial.hermite import hermgauss

class GaussianCopulaPricer:
    def __init__(self, discount_file, constituent_file):
        self.load_data(discount_file, constituent_file)
        self.setup_integration()
        
    def load_data(self, discount_file, constituent_file):
        """Loads curve data and prepares interpolation functions."""
        # 1. Discount Curve
        df_disc = pd.read_csv(discount_file)
        # Log-linear interpolation for Discount Factors
        self.df_interp = interp1d(df_disc['Tenor'], np.log(df_disc['DF']), 
                                  kind='linear', fill_value="extrapolate")
        
        # 2. Constituent Curves (Adjusted)
        self.const_df = pd.read_csv(constituent_file)
        self.recoveries = self.const_df['Recovery'].values
        self.n_constituents = len(self.const_df)
        
        # Prepare survival probability interpolators for each constituent
        # We assume columns are like 'S_1Y', 'S_2Y', etc.
        self.tenor_cols = [c for c in self.const_df.columns if c.startswith('S_') and 'Y' in c]
        tenors = sorted([float(c.replace('S_', '').replace('Y', '')) for c in self.tenor_cols])
        self.max_maturity = max(tenors)
        
        # Create a 3D array for fast lookup: [Company, Time_Index] is difficult due to interpolation needs
        # Instead, we'll interpolate on the fly or pre-grid for the specific tranche pricing
        self.surv_data = self.const_df[self.tenor_cols].values
        self.surv_tenors = np.array(tenors)

    def get_df(self, t):
        """Returns Discount Factor at time t."""
        if t == 0: return 1.0
        return np.exp(self.df_interp(t))

    def get_survival_probs(self, t):
        """Returns array of survival probabilities for all constituents at time t."""
        # Log-linear interpolation in time dimension
        # We do this vectorized for all companies
        
        # Handle t=0
        if t == 0:
            return np.ones(self.n_constituents)
        
        # Find indices for interpolation
        if t >= self.surv_tenors[-1]:
            # Extrapolate flat hazard rate (log-linear survival) from last segment
            idx = len(self.surv_tenors) - 2
        else:
            idx = np.searchsorted(self.surv_tenors, t) - 1
            if idx < 0: idx = 0 # Should not happen if 0 is not in tenors but we handle t<first tenor
            
        t1 = self.surv_tenors[idx]
        t2 = self.surv_tenors[idx+1] if idx + 1 < len(self.surv_tenors) else t1 + 1.0 # Fallback
        
        log_s1 = np.log(self.surv_data[:, idx])
        log_s2 = np.log(self.surv_data[:, idx+1]) if idx + 1 < len(self.surv_tenors) else np.log(self.surv_data[:, idx]) * (t2/t1) # Rough extrapolation
        
        # Interpolate
        frac = (t - t1) / (t2 - t1)
        log_st = log_s1 + frac * (log_s2 - log_s1)
        
        return np.exp(log_st)

    def setup_integration(self):
        """Sets up Gauss-Hermite quadrature nodes and weights for integration over M."""
        # We use 20 points for high accuracy
        degrees = 20
        self.x_nodes, self.weights = hermgauss(degrees)
        # hermgauss integrates exp(-x^2), we need standard normal density
        # Standard Normal M ~ N(0,1).
        # Transform: M = sqrt(2) * x_node
        # Weight adjustment: weight / sqrt(pi)
        self.m_grid = np.sqrt(2) * self.x_nodes
        self.w_grid = self.weights / np.sqrt(np.pi)

    def get_conditional_probs(self, t, rho, m):
        """
        Calculates conditional default probability p_i(t|m).
        p_i(t|m) = Phi( (Phi^-1(P_i(t)) - sqrt(rho)*m) / sqrt(1-rho) )
        """
        # Marginal default prob
        S_t = self.get_survival_probs(t)
        P_t = 1.0 - S_t
        
        # Avoid 0 or 1 for numerical stability in norm.ppf
        P_t = np.clip(P_t, 1e-9, 1.0 - 1e-9)
        
        # Threshold D_i(t)
        thresh = norm.ppf(P_t)
        
        # Conditional Prob
        # If rho is 0, independent
        if rho < 1e-5:
            return P_t
        if rho > 0.999:
            # If rho is 1, default if m < thresh (roughly)
            return (m < thresh).astype(float)
            
        num = thresh - np.sqrt(rho) * m
        den = np.sqrt(1 - rho)
        return norm.cdf(num / den)

    def compute_portfolio_loss_dist(self, probs, recoveries):
        """
        Computes PMF of portfolio loss given default probs and recoveries.
        Assumes equal notional (1/N) for simplicity in recursion.
        Returns: (loss_levels, probability_mass)
        """
        # We use the recursive method (Andersen-Sidenius) for exact distribution
        # Portfolio Notional = 1.0. Loss given default = (1-R) * (1/N)
        
        # Note: If recoveries are different, the lattice becomes complex.
        # We assume constant Recovery = 0.4 for the index structure often used in models,
        # OR we use the average loss given default approx if recoveries vary slightly.
        # For rigorous pricing with different recoveries, we bucket losses.
        
        # Simplified Approach: Assume Loss Amount is standard (1-R_avg)/N per default
        # Or better: The "Loss Unit" approach.
        
        N = self.n_constituents
        loss_unit = (1.0 - 0.4) / N # Base unit assuming 40% Rec
        
        # Initialize PDF: P(Loss=0) = 1.0
        # dp[k] = Probability of having exactly k defaults
        dp = np.zeros(N + 1)
        dp[0] = 1.0
        
        for p in probs:
            # Convolve: New_DP[k] = DP[k]*(1-p) + DP[k-1]*p
            # working backwards to update in place
            dp[1:] = dp[1:] * (1 - p) + dp[:-1] * p
            dp[0] = dp[0] * (1 - p)

        # Map default count to loss amount
        # Loss = k * (1 - 0.4) / N ?? 
        # Actually, if recoveries differ, this is an approximation.
        # For this exercise, we used individual recoveries in curve building.
        # We will calculate Expected Loss rigorously but for the Distribution, 
        # varying recoveries makes the grid non-recombining.
        # APPROXIMATION: We use the individual recoveries to compute Expected Loss,
        # but for the Tranche Thresholds, we map 'k' defaults to 'k' average losses.
        
        # Let's stick to the standard homogeneous recovery assumption for the "Tranche Logic"
        # (Industry standard for simple copula often assumes fixed 40% recovery for tranche mapping)
        avg_rec = 0.4
        loss_levels = np.arange(N + 1) * (1 - avg_rec) / N
        
        return loss_levels, dp

    def price_tranche(self, tranche, rho):
        """
        Prices a CDX tranche.
        
        tranche: dict with keys:
            'Attachment': float (e.g., 0.03)
            'Detachment': float (e.g., 0.07)
            'Maturity': float (e.g., 5.0)
            'Spread_bps': float (Running spread in bps, e.g., 100 or 500)
            'Type': 'Spread' or 'Upfront' (Pricing metric)
        
        rho: Correlation (0 to 1)
        
        Returns:
            float: Model Price (Upfront points or Par Spread bps)
        """
        K_att = tranche['Attachment']
        K_det = tranche['Detachment']
        T = tranche['Maturity']
        coupon_bps = tranche.get('Spread_bps', 100 if K_att > 0 else 500)
        quote_type = tranche.get('Type', 'Spread' if K_att > 0 else 'Upfront')
        
        dt = 0.25
        time_grid = np.arange(dt, T + 0.01, dt)
        
        # Store Unconditional Expected Tranche Losses at each time step
        # E[TL(t)]
        ETL_curve = []
        
        # 1. Calculate Expected Tranche Loss for each time step
        for t in time_grid:
            
            # Integrate over M
            expected_tranche_loss_t = 0.0
            
            for m, w in zip(self.m_grid, self.w_grid):
                # Conditional Probabilities
                cond_probs = self.get_conditional_probs(t, rho, m)
                
                # Portfolio Loss Distribution given M
                loss_levels, pdf_loss = self.compute_portfolio_loss_dist(cond_probs, self.recoveries)
                
                # Tranche Loss for each portfolio loss level L
                # TL = Min(Max(L - K_att, 0), K_det - K_att)
                tranche_losses = np.minimum(np.maximum(loss_levels - K_att, 0), K_det - K_att)
                
                # Conditional Expected Tranche Loss
                cond_ETL = np.sum(tranche_losses * pdf_loss)
                
                expected_tranche_loss_t += cond_ETL * w
            
            ETL_curve.append(expected_tranche_loss_t)
            
        ETL_curve = np.array(ETL_curve)
        tranche_notional = K_det - K_att
        
        # 2. Calculate PV of Legs
        pv_protection = 0.0
        pv_premium = 0.0
        
        prev_etl = 0.0
        prev_t = 0.0
        
        for i, t in enumerate(time_grid):
            curr_etl = ETL_curve[i]
            
            df = self.get_df(t)
            
            # Protection Leg: Pay on increase in EL
            # Approx: Pay at time t
            d_etl = curr_etl - prev_etl
            pv_protection += d_etl * df
            
            # Premium Leg: Pay on outstanding notional
            # Outstanding = Tranche_Width - E[Tranche_Loss]
            # Approx: avg outstanding over period
            avg_outstanding = tranche_notional - (curr_etl + prev_etl) / 2
            dt_actual = t - prev_t
            pv_premium += avg_outstanding * dt_actual * df
            
            prev_etl = curr_etl
            prev_t = t
            
        # 3. Convert to Price
        # Coupon in decimal
        c = coupon_bps / 10000.0
        
        if quote_type == 'Upfront':
            # Upfront = PV_Prot - PV_Prem_at_Fixed_Coupon
            # Usually quoted as percent of notional (0 to 1 or 0 to 100)
            # We return points (0 to 1)
            upfront = (pv_protection - c * pv_premium) / tranche_notional
            return upfront # * 100 for percent
            
        else: # Quote Spread
            # Fair Spread = PV_Prot / PV_Risky_Duration
            if pv_premium < 1e-9: return 0.0
            par_spread = pv_protection / pv_premium
            return par_spread * 10000 # Return in bps

# Wrapper function as requested
def Price_Gaussian(Tranche, rho):
    """
    Standalone wrapper to initialize the pricer and return price.
    Assumes csv files are in local directory.
    """
    pricer = GaussianCopulaPricer('discount_curve.csv', 'adjusted_constituent_survival_curves.csv')
    return pricer.price_tranche(Tranche, rho)

# Example Usage Block
if __name__ == "__main__":
    # Example: Price the 3-7% Tranche at 5Y with 40% correlation
    tranche_3_7 = {
        'Attachment': 0.03,
        'Detachment': 0.07,
        'Maturity': 5.0,
        'Spread_bps': 100, # Standard IG spread
        'Type': 'Spread'
    }
    
    tranche_equity = {
        'Attachment': 0.00,
        'Detachment': 0.03,
        'Maturity': 5.0,
        'Spread_bps': 500,
        'Type': 'Upfront'
    }

    # Need files to run this, wrapping in try-except for robust local testing
    try:
        # Initialize once for speed in a loop
        pricer = GaussianCopulaPricer('discount_curve.csv', 'adjusted_constituent_survival_curves.csv')
        
        price_mz = pricer.price_tranche(tranche_3_7, rho=0.4)
        print(f"3-7% Tranche Spread (rho=0.4): {price_mz:.2f} bps")
        
        price_eq = pricer.price_tranche(tranche_equity, rho=0.4)
        print(f"0-3% Tranche Upfront (rho=0.4): {price_eq*100:.2f}%")
        
    except FileNotFoundError:
        print("Files not found. Please ensure 'discount_curve.csv' and 'adjusted_constituent_survival_curves.csv' exist.")