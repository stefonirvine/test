
import pandas as pd
import pyomo.environ as pe
import pyomo.gdp as pyogdp
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import product


class TheatreScheduler:

    def __init__(self, case_file_path, session_file_path):
        """
        Read case and session data into Pandas DataFrames
        Args:
            case_file_path (str): path to case data in CSV format
            session_file_path (str): path to theatre session data in CSV format
        """
        try:
            self.df_cases = pd.read_csv('cases.csv')
        except FileNotFoundError:
            print("Case data not found.")

        try:
            self.df_sessions = pd.read_csv('sessions.csv')
        except FileNotFoundError:
            print("Session data not found")

        self.model = self.create_model()

    def _generate_case_durations(self):
        """
        Generate mapping of cases IDs to median case time for the procedure
        Returns:
            (dict): dictionary with CaseID as key and median case time (mins) for procedure as value
        """
        return pd.Series(self.df_cases["Median Duration"].values, index=self.df_cases["CaseID"]).to_dict()

    def _generate_session_durations(self):
        """
        Generate mapping of all theatre sessions IDs to session duration in minutes
        Returns:
            (dict): dictionary with SessionID as key and session duration as value
        """
        return pd.Series(self.df_sessions["Duration"].values, index=self.df_sessions["SessionID"]).to_dict()

    def _generate_session_start_times(self):
        """
        Generate mapping from SessionID to session start time
        Returns:
            (dict): dictionary with SessionID as key and start time in minutes since midnight as value
        """
        # Convert session start time from HH:MM:SS format into seconds elapsed since midnight
        self.df_sessions.loc[:, "Start"] = pd.to_timedelta(self.df_sessions["Start"])
        self.df_sessions.loc[:, "Start"] = self.df_sessions["Start"].dt.total_seconds() / 60
        return pd.Series(self.df_sessions["Start"].values, index=self.df_sessions["SessionID"]).to_dict()

    def _get_ordinal_case_deadlines(self):
        """
        Converts the "TargetDeadline" column of the self.df_cases dataframe to ordinal format
        using datetime.toordinal() and returns a dictionary mapping each CaseID to its
        corresponding ordinal value.
        """
        self.df_cases.loc[:, "TargetDeadline"] = pd.to_datetime(self.df_cases["TargetDeadline"], format="%d/%m/%Y")
        self.df_cases.loc[:, "TargetDeadline"] = self.df_cases["TargetDeadline"].apply(lambda date: date.toordinal())
        return dict(zip(self.df_cases["CaseID"], self.df_cases["TargetDeadline"]))


    def _get_ordinal_session_dates(self):
        """
        Converts the "Date" column of the self.df_sessions dataframe to ordinal format
        using datetime.toordinal() and returns a dictionary mapping each SessionID to its
        corresponding ordinal value.
        """
        self.df_sessions.loc[:, "Date"] = pd.to_datetime(self.df_sessions["Date"], format="%d/%m/%Y")
        self.df_sessions.loc[:, "Date"] = self.df_sessions["Date"].apply(lambda date: date.toordinal())
        return dict(zip(self.df_sessions["SessionID"], self.df_sessions["Date"]))


    def _generate_disjunctions(self):
        """
        Generates a list of all disjunctions using the product of all possible pairs of cases and sessions.

        Returns:
            disjunctions (list): list of tuples containing disjunctions
        """
        cases = self.df_cases["CaseID"].to_list()
        sessions = self.df_sessions["SessionID"].to_list()
        disjunctions = list(product(cases, cases, sessions))
        disjunctions = [(case1, case2, session) for (case1, case2, session) in disjunctions if (case1 != case2)]
        return disjunctions

    def create_model(self):
        model = pe.ConcreteModel()

        # Model Data

        # List of case IDs in surgical waiting list
        model.CASES = pe.Set(initialize=self.df_cases["CaseID"].tolist())
        # List of sessions IDs
        model.SESSIONS = pe.Set(initialize=self.df_sessions["SessionID"].tolist())
        # List of tasks - all possible (caseID, sessionID) combination
        model.TASKS = pe.Set(initialize=model.CASES * model.SESSIONS, dimen=2)
        # The duration (median case time) for each operation
        model.CASE_DURATION = pe.Param(model.CASES, initialize=self._generate_case_durations())
        # The duration of each theatre session
        model.SESSION_DURATION = pe.Param(model.SESSIONS, initialize=self._generate_session_durations())
        # The start time of each theatre session
        model.SESSION_START_TIME = pe.Param(model.SESSIONS, initialize=self._generate_session_start_times())
        # The deadline of each case
        model.CASE_DEADLINES = pe.Param(model.CASES, initialize=self._get_ordinal_case_deadlines())
        # The date of each theatre session
        model.SESSION_DATES = pe.Param(model.SESSIONS, initialize=self._get_ordinal_session_dates())

        model.DISJUNCTIONS = pe.Set(initialize=self._generate_disjunctions(), dimen=3)

        ub = 1440  # seconds in a day
        model.M = pe.Param(initialize=1e3*ub)  # big M
        max_util = 0.90
        num_cases = self.df_cases.shape[0]

        # Decision Variables
        model.SESSION_ASSIGNED = pe.Var(model.TASKS, domain=pe.Binary)
        model.CASE_START_TIME = pe.Var(model.TASKS, bounds=(0, ub), within=pe.PositiveReals)
        model.CASES_IN_SESSION = pe.Var(model.SESSIONS, bounds=(0, num_cases), within=pe.PositiveReals)
        model.max_util = pe.Var(within=pe.NonNegativeReals)

        # Objective
        def objective_function(model):
            return pe.summation(model.CASES_IN_SESSION)
            return sum([model.SESSION_ASSIGNED[case, session] for case in model.CASES for session in model.SESSIONS])
        model.OBJECTIVE = pe.Objective(rule=objective_function, sense=pe.maximize)

        # Constraints

        # Case start time must be after start time of assigned theatre session
        def case_start_time(model, case, session):
            return model.CASE_START_TIME[case, session] >= model.SESSION_START_TIME[session] - \
                   ((1 - model.SESSION_ASSIGNED[(case, session)])*model.M)
        model.CASE_START = pe.Constraint(model.TASKS, rule=case_start_time)

        # Case end time must be before end time of assigned theatre session
        def case_end_time(model, case, session):
            return model.CASE_START_TIME[case, session] + model.CASE_DURATION[case] <= model.SESSION_START_TIME[session] + \
                   model.SESSION_DURATION[session]*max_util + ((1 - model.SESSION_ASSIGNED[(case, session)]) * model.M)
        model.CASE_END_TIME = pe.Constraint(model.TASKS, rule=case_end_time)

        # Cases can be assigned to a maximum of one session
        def session_assignment(model, case):
            return sum([model.SESSION_ASSIGNED[(case, session)] for session in model.SESSIONS]) <= 1
        model.SESSION_ASSIGNMENT = pe.Constraint(model.CASES, rule=session_assignment)

        def set_deadline_condition(model, case, session):
            return model.SESSION_DATES[session] <= model.CASE_DEADLINES[case] + ((1 - model.SESSION_ASSIGNED[case, session])*model.M)
        model.APPLY_DEADLINE = pe.Constraint(model.TASKS, rule=set_deadline_condition)

        def no_case_overlap(model, case1, case2, session):
            return [model.CASE_START_TIME[case1, session] + model.CASE_DURATION[case1] <= model.CASE_START_TIME[case2, session] + \
                    ((2 - model.SESSION_ASSIGNED[case1, session] - model.SESSION_ASSIGNED[case2, session])*model.M),
                    model.CASE_START_TIME[case2, session] + model.CASE_DURATION[case2] <= model.CASE_START_TIME[case1, session] + \
                    ((2 - model.SESSION_ASSIGNED[case1, session] - model.SESSION_ASSIGNED[case2, session])*model.M)]

        model.DISJUNCTIONS_RULE = pyogdp.Disjunction(model.DISJUNCTIONS, rule=no_case_overlap)

        def theatre_util(model, session):
            return model.CASES_IN_SESSION[session] == \
                   sum([model.SESSION_ASSIGNED[case, session] for case in model.CASES])

        model.THEATRE_UTIL = pe.Constraint(model.SESSIONS, rule=theatre_util)

        pe.TransformationFactory("gdp.bigm").apply_to(model)

        return model

    def solve(self, solver_name, options=None, solver_path=None, local=True):

        if solver_path is not None:
            solver = pe.SolverFactory(solver_name, executable=solver_path)
        else:
            solver = pe.SolverFactory(solver_name)

        # TODO remove - too similar to alstom
        if options is not None:
            for key, value in options.items():
                solver.options[key] = value

        if local:
            solver_results = solver.solve(self.model, tee=True)
        else:
            solver_manager = pe.SolverManagerFactory("neos")
            solver_results = solver_manager.solve(self.model, opt=solver)

        results = [{"Case": case,
                    "Session": session,
                    "Session Date": self.model.SESSION_DATES[session],
                    "Case Deadline": self.model.CASE_DEADLINES[case],
                    "Days before deadline": self.model.CASE_DEADLINES[case] - self.model.SESSION_DATES[session],
                    "Start": self.model.CASE_START_TIME[case, session](),
                    "Assignment": self.model.SESSION_ASSIGNED[case, session]()}
                   for (case, session) in self.model.TASKS]

        self.df_times = pd.DataFrame(results)

        all_cases = self.model.CASES.ordered_data()
        cases_assigned = []
        for (case, session) in self.model.SESSION_ASSIGNED:
            if self.model.SESSION_ASSIGNED[case, session].is_fixed() and self.model.SESSION_ASSIGNED[case, session].value == 1:
                cases_assigned.append(case)

        cases_missed = list(set(all_cases).difference(cases_assigned))
        print("Number of cases assigned = {} out of {}:".format(sum(self.model.CASES_IN_SESSION.get_values().values()), len(all_cases)))
        print("Cases assigned: ", cases_assigned)
        print("Number of cases missed = {} out of {}:".format(len(cases_missed), len(all_cases)))
        print("Cases missed: ", cases_missed)
        self.model.CASES_IN_SESSION.pprint()
        print("Total Objective = {}".format(sum(self.model.CASES_IN_SESSION.get_values().values())))
        print("Number of constraints = {}".format(solver_results["Problem"].__getitem__(0)["Number of constraints"]))
        #self.model.SESSION_ASSIGNED.pprint()
        print(self.df_times[self.df_times["Assignment"] == 1].to_string())
        self.draw_gantt()

    def draw_gantt(self):

        # Filter the dataframe to only include assigned cases
        df = self.df_times[self.df_times["Assignment"] == 1]

        # Get the unique cases and sessions
        cases = sorted(list(df['Case'].unique()))
        sessions = sorted(list(df['Session'].unique()))

        # Define styles for the bars and text labels
        bar_style = {'alpha': 1.0, 'lw': 25, 'solid_capstyle': 'butt'}
        text_style = {'color': 'white', 'weight': 'bold', 'ha': 'center', 'va': 'center'}

        # Define colors using the Dark2 colormap from matplotlib
        colors = cm.Dark2.colors

        # Sort the dataframe by case and session, then set the index
        df.sort_values(by=['Case', 'Session'], inplace=True)
        df.set_index(['Case', 'Session'], inplace=True)

        # Create a figure and axis object
        fig, ax = plt.subplots(1, 1)

        # Loop through each case and session
        for c_ix, c in enumerate(cases, 1):
            for s_ix, s in enumerate(sessions, 1):
                if (c, s) in df.index:
                    # Get the start time and duration for the case
                    xs = df.loc[(c, s), 'Start']
                    xf = xs + self.df_cases.loc[self.df_cases["CaseID"] == c, "Median Duration"].iloc[0]

                    # Plot the bar and add the case label
                    ax.plot([xs, xf], [s] * 2, c=colors[c_ix % len(colors)], **bar_style)
                    ax.text((xs + xf) / 2, s, c, **text_style)

        # Set the plot title, x-axis label, and y-axis label
        ax.set_title('Assigning Cases to Theatre Sessions')
        ax.set_xlabel('Time')
        ax.set_ylabel('Sessions')

        # Show the grid
        ax.grid(True)

        # Adjust the plot layout to prevent labels from getting cut off
        fig.tight_layout()

        # Show the plot
        plt.show()


if __name__ == "__main__":
    case_path = os.path.join(os.path.dirname(os.getcwd()), "data", "cases.csv")
    session_path = os.path.join(os.path.dirname(os.getcwd()), "data", "sessions.csv")
    cbc_path = "/usr/local/opt/cbc/bin/cbc"

    options = {"seconds": 50}
    scheduler = TheatreScheduler(case_file_path=case_path, session_file_path=session_path)
    scheduler.solve(solver_name="cbc", solver_path=cbc_path, options=options)
    #scheduler.solve(solver_name="cbc", local=False, options=None)

