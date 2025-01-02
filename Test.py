import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st


class DiversifiedPortfolioSimulator:
    def __init__(self, tickers, weights, initial_value, start_date, end_date, time_horizon, n_simulations):
        self.tickers = tickers
        self.weights = np.array(weights)
        self.initial_value = initial_value
        self.start_date = start_date
        self.end_date = end_date
        self.time_horizon = time_horizon
        self.n_simulations = n_simulations
        self.price_data = None
        self.returns_data = None
        self.mean_returns = None
        self.cov_matrix = None
        
    def ensure_positive_semidefinite(matrix, epsilon=1e-8):
        try:
            # Compute the eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            
            # Replace negative eigenvalues with a small positive value
            eigenvalues = np.maximum(eigenvalues, epsilon)
            
            # Reconstruct the matrix
            psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            return psd_matrix
        except np.linalg.LinAlgError as e:
            raise ValueError("Failed to make the covariance matrix positive semidefinite.") from e

    def fetch_data(self):
        st.write("Fetching historical price data for diversified portfolio...")
        try:
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
        except KeyError:
            raise ValueError("Could not fetch data. Check ticker symbols or date range.")

        if data.empty:
            raise ValueError("No data retrieved. Please check the tickers or date range.")
        if len(data.columns) != len(self.tickers):
            st.warning("Some tickers returned no data. They will be excluded.")
            data = data.dropna(axis=1, how="all")

        self.price_data = data
        self.returns_data = self.price_data.pct_change().dropna()
        self.mean_returns = self.returns_data.mean().values
        self.cov_matrix = self.returns_data.cov().values
        st.write("Data successfully fetched and processed.")

    def simulate_portfolio(self):
        st.write("Running Monte Carlo simulations...")
        simulated_returns = np.random.multivariate_normal(
            self.mean_returns, self.cov_matrix, (self.time_horizon, self.n_simulations)
            )
        portfolio_returns = np.dot(simulated_returns, self.weights)
        portfolio_values = self.initial_value * (1 + portfolio_returns).cumprod(axis=0)
        return portfolio_values

        # multivariate_normal =  This function generates random returns for each asset in the portfolio based on a multivariate normal distribution.
        # mean_returns=The average return you expect for each asset (like Mutual funds, Crypto etc..,).
        # cov_matrix = The covariance matrix, which tells how the returns of each asset are related to each other.
        # time_horizon, n_simulations): This specifies the number of time steps (e.g., number of days) and the number of simulations you want to run. For example, 252 days (time_horizon) and 1000 simulations (n_simulations).
        # np.dot: This calculates the dot product between the simulated returns and the portfolio weights.
        
        # (1 + portfolio_returns): This converts the portfolio returns into the growth factor (e.g., a return of 10% becomes 1.10, which means your portfolio grows by 10%).
        # .cumprod(axis=0): This calculates the cumulative product of the growth factors over time for each simulation.
        # 

    def analyze_simulation(self, portfolio_values):
        st.write("Analyzing simulation results...")
        final_values = portfolio_values[-1, :]
        # 
        # portfolio_values[-1, :]: This accesses the portfolio values from the last time step (i.e., the final day of the simulation) for all simulations.
        # [-1]: Refers to the last row (final day) of the portfolio_values array (the time dimension).
        # [:]: Refers to all columns, which represent the different simulations.
        # 
        mean_final_value = np.mean(final_values) 
        VaR_95 = np.percentile(final_values, 5) #Value at Risk (VaR) is a risk measure that tells you the potential loss you might face under the worst 5% of scenarios, with a 95% confidence level.
        max_loss = self.initial_value - np.min(final_values) #np.min(final_values): This gives you the minimum final portfolio value across all simulations (the worst-case final value).
        sharpe_ratio = (mean_final_value - self.initial_value) / np.std(final_values)

        results = {
            "Mean Final Value": mean_final_value,
            "Value at Risk (95%)": VaR_95,
            "Maximum Loss": max_loss,
            "Sharpe Ratio": sharpe_ratio,
        }
        return results

    def plot_growth_paths(self, portfolio_values):
        st.write("Simulated Portfolio Growth Paths")
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_values[:, :100], alpha=0.05, color="blue")
        plt.title("Monte Carlo Simulations: Portfolio Growth")
        plt.xlabel("Days")
        plt.ylabel("Portfolio Value")
        st.pyplot(plt)

    def plot_final_value_distribution(self, portfolio_values):
        st.write("Final Portfolio Value Distribution")
        final_values = portfolio_values[-1, :]
        VaR_95 = np.percentile(final_values, 5)

        plt.figure(figsize=(10, 6))
        plt.hist(final_values, bins=50, alpha=0.7, color="blue", label="Final Portfolio Values")
        plt.axvline(VaR_95, color='red', linestyle='dashed', linewidth=2, label="VaR (95%)")
        plt.title("Distribution of Final Portfolio Values")
        plt.xlabel("Portfolio Value")
        plt.ylabel("Frequency")
        plt.legend()
        st.pyplot(plt)

    def run(self):
        try:
            self.fetch_data()
            portfolio_values = self.simulate_portfolio()
            results = self.analyze_simulation(portfolio_values)

            st.write("Simulation Results:")
            for key, value in results.items():
                st.write(f"{key}: ${value:,.2f}" if "Value" in key or "Loss" in key else f"{key}: {value:.2f}")

            self.plot_growth_paths(portfolio_values)
            self.plot_final_value_distribution(portfolio_values)
        except Exception as e:
            st.error(f"An error occurred: {e}")


# Streamlit Interface
def main():
    st.title("Monte Carlo Simulation for Diversified Portfolios")
    st.write("Simulate and analyze portfolio performance for mutual funds, crypto, bonds, and precious metals.")

    # User Inputs
    #tickers = ["VTSMX", "BTC-USD", "IEF", "GLD"]  # Mutual Fund, Crypto, Bond ETF, Precious Metals
    tickers = st.text_input("Enter tickers (comma-separated)", "VTSMX,BTC-USD,IEF,GLD").split(",")
    # Risk Appetite Dropdown
    risk_appetite = st.selectbox("Select Risk Appetite", ["Conservative", "Moderate", "Aggressive"])
    
    # Adjust weights based on risk appetite
    if risk_appetite == "Conservative":
        weights = [0.2, 0.1, 0.5, 0.2]  # Low risk: Bonds and Precious Metals dominate
    elif risk_appetite == "Moderate":
        weights = [0.3, 0.2, 0.3, 0.2]  # Balanced: Equal exposure
    elif risk_appetite == "Aggressive":
        weights = [0.4, 0.4, 0.1, 0.1]  # High risk: Crypto and Mutual Funds dominate

    # Show adjusted weights
    st.write(f"Adjusted Weights based on '{risk_appetite}' Risk Appetite:")
    st.write(f"Mutual Funds: {weights[0]:.2f}, Crypto: {weights[1]:.2f}, Bonds: {weights[2]:.2f}, Precious Metals: {weights[3]:.2f}")

    # Other Inputs
    initial_value = st.number_input("Initial Portfolio Value ($)", min_value=1000, value=100000)
    start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    time_horizon = st.number_input("Time Horizon (Days)", min_value=1, value=252)
    n_simulations = st.number_input("Number of Simulations", min_value=1, value=10000)

    if st.button("Run Simulation"):
        simulator = DiversifiedPortfolioSimulator(
            tickers, weights, initial_value, start_date, end_date, time_horizon, n_simulations
        )
        simulator.run()


# Run the App
if __name__ == "__main__":
    main()
