import streamlit as st
import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
import io
import base64
from multiprocessing import Pool
import warnings

warnings.filterwarnings("ignore")

class LBOModel:
    """
    A class to model Leveraged Buyout (LBO) financial projections and returns.
    """

    def __init__(self, initial_revenue, ebitda_margin, revenue_growth,
                 tax_rate, capex_percent, nwc_percent,
                 entry_multiple, exit_multiple,
                 debt_to_ebitda, interest_rate, loan_term, amortization_rate,
                 projection_years=5):
        """
        Initialize the LBO model with given parameters.

        Args:
            initial_revenue (float): Initial revenue in millions.
            ebitda_margin (float): EBITDA margin as a decimal.
            revenue_growth (float): Annual revenue growth rate as a decimal.
            tax_rate (float): Tax rate as a decimal.
            capex_percent (float): Capital expenditures as a percentage of revenue.
            nwc_percent (float): Net working capital as a percentage of revenue.
            entry_multiple (float): Entry EV/EBITDA multiple.
            exit_multiple (float): Exit EV/EBITDA multiple.
            debt_to_ebitda (float): Debt to EBITDA ratio.
            interest_rate (float): Interest rate on debt.
            loan_term (int): Loan term in years.
            amortization_rate (float): Annual amortization rate of debt.
            projection_years (int): Number of years to project financials.
        """
        self.initial_revenue = initial_revenue
        self.ebitda_margin = ebitda_margin
        self.revenue_growth = revenue_growth
        self.tax_rate = tax_rate
        self.capex_percent = capex_percent
        self.nwc_percent = nwc_percent
        self.entry_multiple = entry_multiple
        self.exit_multiple = exit_multiple
        self.debt_to_ebitda = debt_to_ebitda
        self.interest_rate = interest_rate
        self.loan_term = loan_term
        self.amortization_rate = amortization_rate
        self.projection_years = projection_years

    def project_financials(self):
        """
        Projects the company's financials over the projection period.

        Returns:
            pd.DataFrame: Financial projections including revenue, EBITDA, FCF, etc.
        """
        df = pd.DataFrame({'Year': range(self.projection_years + 1)})
        df['Revenue'] = self.initial_revenue * (1 + self.revenue_growth) ** df['Year']
        df['EBITDA'] = df['Revenue'] * self.ebitda_margin
        df['Depreciation'] = df['Revenue'] * self.capex_percent
        df['EBIT'] = df['EBITDA'] - df['Depreciation']
        df['Taxes'] = df['EBIT'].clip(lower=0) * self.tax_rate
        df['NOPAT'] = df['EBIT'] - df['Taxes']
        df['Capital Expenditures'] = df['Revenue'] * self.capex_percent
        df['Change in NWC'] = df['Revenue'].diff().fillna(0) * self.nwc_percent
        df['Free Cash Flow'] = df['NOPAT'] + df['Depreciation'] - df['Capital Expenditures'] - df['Change in NWC']
        return df

    def calculate_debt_schedule(self):
        """
        Calculates the debt schedule over the projection period.

        Returns:
            pd.DataFrame: Debt schedule including interest expense, principal payment, etc.
        """
        initial_debt = self.debt_to_ebitda * (self.initial_revenue * self.ebitda_margin)
        df = pd.DataFrame({'Year': range(self.projection_years + 1)})
        df['Beginning Balance'] = 0.0
        df.loc[0, 'Beginning Balance'] = initial_debt
        df['Interest Expense'] = 0.0
        df['Principal Payment'] = 0.0
        df['Ending Balance'] = 0.0

        for year in df['Year'][1:]:
            prev_balance = df.loc[year - 1, 'Beginning Balance']
            interest = prev_balance * self.interest_rate
            principal = min(prev_balance * self.amortization_rate, prev_balance)
            ending_balance = prev_balance - principal

            df.loc[year, 'Beginning Balance'] = prev_balance
            df.loc[year, 'Interest Expense'] = interest
            df.loc[year, 'Principal Payment'] = principal
            df.loc[year, 'Ending Balance'] = ending_balance

        return df

    def calculate_returns(self):
        """
        Calculates the IRR and MOIC for the LBO transaction.

        Returns:
            tuple: (IRR, MOIC)
        """
        financials = self.project_financials()
        debt_schedule = self.calculate_debt_schedule()

        enterprise_value = financials.loc[0, 'EBITDA'] * self.entry_multiple
        initial_debt = debt_schedule.loc[0, 'Beginning Balance']
        initial_equity = enterprise_value - initial_debt

        exit_ebitda = financials['EBITDA'].iloc[-1]
        exit_enterprise_value = exit_ebitda * self.exit_multiple
        exit_debt = debt_schedule['Ending Balance'].iloc[-1]
        exit_equity = exit_enterprise_value - exit_debt

        equity_invested = initial_equity
        equity_returned = exit_equity

        cash_flows = [-equity_invested] + [0]*(self.projection_years - 1) + [equity_returned]
        irr = npf.irr(cash_flows)
        moic = equity_returned / equity_invested

        return irr, moic

def generate_random_parameters(lbo_params, simulation_params, num_simulations):
    """
    Generates random parameters for the Monte Carlo simulation based on specified distributions.

    Args:
        lbo_params (dict): Base LBO parameters.
        simulation_params (dict): Parameters to simulate with their distributions.
        num_simulations (int): Number of simulations to run.

    Returns:
        list: A list of dictionaries with random parameters for each simulation.
    """
    param_list = []
    for _ in range(num_simulations):
        params = lbo_params.copy()
        for param, dist_info in simulation_params.items():
            dist_type = dist_info['distribution']
            if dist_type == 'normal':
                mean = dist_info['mean']
                std = dist_info['std']
                value = np.random.normal(mean, std)
            elif dist_type == 'lognormal':
                mean = dist_info['mean']
                std = dist_info['std']
                # Adjust mean and std to parameters of lognormal
                sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
                mu = np.log(mean) - 0.5 * sigma ** 2
                value = np.random.lognormal(mean=mu, sigma=sigma)
            elif dist_type == 'uniform':
                low = dist_info['low']
                high = dist_info['high']
                value = np.random.uniform(low, high)
            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")
            params[param] = value
        param_list.append(params)
    return param_list

def run_single_simulation(params):
    """
    Runs a single LBO simulation with given parameters.

    Args:
        params (dict): Parameters for the LBOModel.

    Returns:
        tuple: (IRR, MOIC)
    """
    try:
        model = LBOModel(**params)
        irr, moic = model.calculate_returns()
        return irr, moic
    except Exception:
        # Handle exceptions such as invalid inputs leading to calculation errors
        return np.nan, np.nan

def run_monte_carlo(num_simulations, lbo_params, simulation_params):
    """
    Runs the Monte Carlo simulation for the LBO model.

    Args:
        num_simulations (int): Number of simulations to run.
        lbo_params (dict): Base LBO parameters.
        simulation_params (dict): Parameters to simulate with their distributions.

    Returns:
        dict: Results of the simulation containing IRRs and MOICs.
    """
    params_list = generate_random_parameters(lbo_params, simulation_params, num_simulations)
    irr_results = []
    moic_results = []

    # Use multiprocessing Pool for parallel processing
    with Pool() as pool:
        results = pool.map(run_single_simulation, params_list)

    irr_results, moic_results = zip(*results)
    irr_results = np.array(irr_results)
    moic_results = np.array(moic_results)

    return {'IRR': irr_results, 'MOIC': moic_results}

def plot_distribution(data, title):
    """
    Plots the distribution of data.

    Args:
        data (array-like): Data to plot.
        title (str): Title of the plot.

    Returns:
        plotly.graph_objs._figure.Figure: Plotly figure.
    """
    mean_value = np.mean(data)
    median_value = np.median(data)

    fig = go.Figure()

    # Add histogram trace
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=30,
        name='Distribution',
        marker_color='blue',
        opacity=0.7
    ))

    # Add mean line
    fig.add_vline(x=mean_value, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_value:.2%}", 
                  annotation_position="top right")

    # Add median line
    fig.add_vline(x=median_value, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_value:.2%}", 
                  annotation_position="top left")

    # Update layout
    fig.update_layout(
        title_text=f'{title} Distribution',
        xaxis_title=title,
        yaxis_title='Frequency',
        showlegend=False,
    )

    return fig

def calculate_probability_distribution(data, bins=10):
    """
    Calculate the probability distribution of the data.

    Args:
        data (array-like): Data to calculate distribution.
        bins (int): Number of bins.

    Returns:
        pd.DataFrame: Probability distribution DataFrame.
    """
    counts, bin_edges = np.histogram(data, bins=bins)
    probabilities = counts / len(data)
    bin_labels = [f"{bin_edges[i]:.2%} to {bin_edges[i+1]:.2%}" for i in range(len(bin_edges)-1)]
    return pd.DataFrame({'Range': bin_labels, 'Probability': probabilities})

def get_excel_download_link(results):
    """
    Generates a link to download results in Excel format.

    Args:
        results (dict): Dictionary containing 'IRR' and 'MOIC' results.

    Returns:
        str: HTML anchor tag with download link.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pd.DataFrame(results['IRR'], columns=['IRR']).to_excel(writer, sheet_name="IRR", index=False)
        pd.DataFrame(results['MOIC'], columns=['MOIC']).to_excel(writer, sheet_name="MOIC", index=False)
    output.seek(0)
    b64 = base64.b64encode(output.read()).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="lbo_simulation_results.xlsx">Download Excel File</a>'

def validate_inputs(lbo_params):
    """
    Validates the input parameters.

    Args:
        lbo_params (dict): LBO parameters to validate.

    Returns:
        list: List of error messages.
    """
    errors = []
    if not (0 < lbo_params['ebitda_margin'] < 1):
        errors.append("EBITDA Margin must be between 0 and 1.")
    if not (0 < lbo_params['tax_rate'] < 1):
        errors.append("Tax Rate must be between 0 and 1.")
    if not (0 < lbo_params['capex_percent'] < 1):
        errors.append("CapEx Percent must be between 0 and 1.")
    if not (0 < lbo_params['nwc_percent'] < 1):
        errors.append("Net Working Capital Percent must be between 0 and 1.")
    if lbo_params['debt_to_ebitda'] <= 0:
        errors.append("Debt to EBITDA must be greater than 0.")
    if lbo_params['interest_rate'] <= 0 or lbo_params['interest_rate'] >= 1:
        errors.append("Interest Rate must be between 0 and 1.")
    if lbo_params['loan_term'] <= 0:
        errors.append("Loan Term must be greater than 0.")
    if not (0 < lbo_params['amortization_rate'] < 1):
        errors.append("Amortization Rate must be between 0 and 1.")
    if lbo_params['projection_years'] <= 0:
        errors.append("Projection Years must be greater than 0.")
    return errors

def main():
    st.set_page_config(page_title="Advanced LBO Monte Carlo Simulation", layout="wide")
    st.title("Advanced LBO Monte Carlo Simulation")

    st.sidebar.header("LBO Model Parameters")
    initial_revenue = st.sidebar.number_input("Initial Revenue ($M)", value=100.0, step=10.0)
    ebitda_margin = st.sidebar.slider("EBITDA Margin", 0.05, 0.5, 0.2, 0.01)
    revenue_growth = st.sidebar.slider("Revenue Growth Rate", 0.01, 0.2, 0.05, 0.01)
    tax_rate = st.sidebar.slider("Tax Rate", 0.1, 0.4, 0.25, 0.01)
    capex_percent = st.sidebar.slider("CapEx (% of Revenue)", 0.01, 0.1, 0.05, 0.01)
    nwc_percent = st.sidebar.slider("Net Working Capital (% of Revenue)", 0.05, 0.2, 0.1, 0.01)
    entry_multiple = st.sidebar.slider("Entry Multiple", 5.0, 15.0, 8.0, 0.5)
    exit_multiple = st.sidebar.slider("Exit Multiple", 5.0, 15.0, 10.0, 0.5)
    debt_to_ebitda = st.sidebar.slider("Debt to EBITDA", 3.0, 7.0, 5.0, 0.5)
    interest_rate = st.sidebar.slider("Interest Rate", 0.03, 0.1, 0.06, 0.005)
    loan_term = st.sidebar.slider("Loan Term (Years)", 5, 10, 7, 1)
    amortization_rate = st.sidebar.slider("Annual Amortization Rate", 0.05, 0.2, 0.1, 0.01)
    projection_years = st.sidebar.slider("Projection Years", 3, 10, 5, 1)

    lbo_params = {
        'initial_revenue': initial_revenue,
        'ebitda_margin': ebitda_margin,
        'revenue_growth': revenue_growth,
        'tax_rate': tax_rate,
        'capex_percent': capex_percent,
        'nwc_percent': nwc_percent,
        'entry_multiple': entry_multiple,
        'exit_multiple': exit_multiple,
        'debt_to_ebitda': debt_to_ebitda,
        'interest_rate': interest_rate,
        'loan_term': loan_term,
        'amortization_rate': amortization_rate,
        'projection_years': projection_years
    }

    st.sidebar.header("Monte Carlo Simulation Parameters")
    num_simulations = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000, step=100)

    st.sidebar.write("### Simulation Distributions")
    simulation_params = {}

    st.sidebar.write("#### Revenue Growth Rate")
    rev_growth_dist = st.sidebar.selectbox("Distribution", ["Normal", "Uniform"], key='rev_growth_dist')
    if rev_growth_dist == "Normal":
        rev_growth_std = st.sidebar.slider("Std Dev", 0.01, 0.1, 0.02, 0.005, key='rev_growth_std')
        simulation_params['revenue_growth'] = {
            'distribution': 'normal',
            'mean': revenue_growth,
            'std': rev_growth_std
        }
    elif rev_growth_dist == "Uniform":
        rev_growth_low = st.sidebar.number_input("Low", value=revenue_growth - 0.02, step=0.005, key='rev_growth_low')
        rev_growth_high = st.sidebar.number_input("High", value=revenue_growth + 0.02, step=0.005, key='rev_growth_high')
        simulation_params['revenue_growth'] = {
            'distribution': 'uniform',
            'low': rev_growth_low,
            'high': rev_growth_high
        }

    st.sidebar.write("#### EBITDA Margin")
    ebitda_margin_dist = st.sidebar.selectbox("Distribution", ["Normal", "Uniform"], key='ebitda_margin_dist')
    if ebitda_margin_dist == "Normal":
        ebitda_margin_std = st.sidebar.slider("Std Dev", 0.01, 0.1, 0.02, 0.005, key='ebitda_margin_std')
        simulation_params['ebitda_margin'] = {
            'distribution': 'normal',
            'mean': ebitda_margin,
            'std': ebitda_margin_std
        }
    elif ebitda_margin_dist == "Uniform":
        ebitda_margin_low = st.sidebar.number_input("Low", value=ebitda_margin - 0.02, step=0.005, key='ebitda_margin_low')
        ebitda_margin_high = st.sidebar.number_input("High", value=ebitda_margin + 0.02, step=0.005, key='ebitda_margin_high')
        simulation_params['ebitda_margin'] = {
            'distribution': 'uniform',
            'low': ebitda_margin_low,
            'high': ebitda_margin_high
        }

    st.sidebar.write("#### Exit Multiple")
    exit_multiple_dist = st.sidebar.selectbox("Distribution", ["Normal", "Lognormal", "Uniform"], key='exit_multiple_dist')
    if exit_multiple_dist == "Normal":
        exit_multiple_std = st.sidebar.slider("Std Dev", 0.5, 3.0, 1.0, 0.1, key='exit_multiple_std')
        simulation_params['exit_multiple'] = {
            'distribution': 'normal',
            'mean': exit_multiple,
            'std': exit_multiple_std
        }
    elif exit_multiple_dist == "Lognormal":
        exit_multiple_std = st.sidebar.slider("Std Dev", 0.5, 3.0, 1.0, 0.1, key='exit_multiple_lognormal_std')
        simulation_params['exit_multiple'] = {
            'distribution': 'lognormal',
            'mean': exit_multiple,
            'std': exit_multiple_std
        }
    elif exit_multiple_dist == "Uniform":
        exit_multiple_low = st.sidebar.number_input("Low", value=exit_multiple - 1.0, step=0.1, key='exit_multiple_low')
        exit_multiple_high = st.sidebar.number_input("High", value=exit_multiple + 1.0, step=0.1, key='exit_multiple_high')
        simulation_params['exit_multiple'] = {
            'distribution': 'uniform',
            'low': exit_multiple_low,
            'high': exit_multiple_high
        }

    st.sidebar.write("#### Debt to EBITDA")
    debt_to_ebitda_dist = st.sidebar.selectbox("Distribution", ["Normal", "Uniform"], key='debt_to_ebitda_dist')
    if debt_to_ebitda_dist == "Normal":
        debt_to_ebitda_std = st.sidebar.slider("Std Dev", 0.1, 1.0, 0.2, 0.05, key='debt_to_ebitda_std')
        simulation_params['debt_to_ebitda'] = {
            'distribution': 'normal',
            'mean': debt_to_ebitda,
            'std': debt_to_ebitda_std
        }
    elif debt_to_ebitda_dist == "Uniform":
        debt_to_ebitda_low = st.sidebar.number_input("Low", value=debt_to_ebitda - 0.5, step=0.1, key='debt_to_ebitda_low')
        debt_to_ebitda_high = st.sidebar.number_input("High", value=debt_to_ebitda + 0.5, step=0.1, key='debt_to_ebitda_high')
        simulation_params['debt_to_ebitda'] = {
            'distribution': 'uniform',
            'low': debt_to_ebitda_low,
            'high': debt_to_ebitda_high
        }

    # Validate inputs
    errors = validate_inputs(lbo_params)
    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    if st.button("Run Simulation"):
        st.write("## Simulation Results")
        progress_text = st.empty()
        progress_bar = st.progress(0)
        results = run_monte_carlo(num_simulations, lbo_params, simulation_params)
        progress_bar.progress(100)
        progress_text.text("Simulation complete.")

        # Remove NaN values from results
        irr_results = results['IRR'][~np.isnan(results['IRR'])]
        moic_results = results['MOIC'][~np.isnan(results['MOIC'])]

        # Display IRR Distribution
        st.write("### Internal Rate of Return (IRR) Distribution")
        fig_irr = plot_distribution(irr_results, 'IRR')
        st.plotly_chart(fig_irr, use_container_width=True)

        # Display MOIC Distribution
        st.write("### Multiple of Invested Capital (MOIC) Distribution")
        fig_moic = plot_distribution(moic_results, 'MOIC')
        st.plotly_chart(fig_moic, use_container_width=True)

        # Statistics and Probability Distributions
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### IRR Statistics")
            df_irr_stats = pd.DataFrame(irr_results, columns=['IRR'])
            st.dataframe(df_irr_stats.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).transpose().style.format("{:.2%}"))

            st.write("#### IRR Probability Distribution")
            irr_prob_dist = calculate_probability_distribution(irr_results, bins=10)
            st.dataframe(irr_prob_dist.style.format({'Probability': '{:.2%}'}))

        with col2:
            st.write("#### MOIC Statistics")
            df_moic_stats = pd.DataFrame(moic_results, columns=['MOIC'])
            st.dataframe(df_moic_stats.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).transpose().style.format("{:.2f}"))

            st.write("#### MOIC Probability Distribution")
            moic_prob_dist = calculate_probability_distribution(moic_results, bins=10)
            st.dataframe(moic_prob_dist.style.format({'Probability': '{:.2%}'}))

        # Export Results
        if st.button("Export Results"):
            st.markdown(get_excel_download_link({'IRR': irr_results, 'MOIC': moic_results}), unsafe_allow_html=True)

        # Display sample financial projections and debt schedule
        st.write("## Sample Financial Projections")
        base_model = LBOModel(**lbo_params)
        financials = base_model.project_financials()
        st.dataframe(financials.style.format({col: '{:,.2f}' for col in financials.columns if col != 'Year'}))

        st.write("## Sample Debt Schedule")
        debt_schedule = base_model.calculate_debt_schedule()
        st.dataframe(debt_schedule.style.format({col: '{:,.2f}' for col in debt_schedule.columns if col != 'Year'}))

if __name__ == "__main__":
    main()
