"""
Copyright (c) 2021 Tilo van Ekeris / TMDT, University of Wuppertal
Distributed under the MIT license, see the accompanying
file LICENSE or https://opensource.org/licenses/MIT
"""

import win32com.client as win32

from .error_code import ErrorCode
from .attribute_explorer import AttributeExplorer
from .plant_simulation_connection import PlantSimulationConnection
from .table import Table


class Plantsim:

    def __init__(self, path_context, model, version='', socket=None, visible=True, trust_models=False,
                 license_type='Professional'):
        print(f'\n+++ Launching Plant Simulation (version={version}, license={license_type}, visibility={visible}, trust_models={trust_models}) +++\n')

        dispatch_string = 'Tecnomatix.PlantSimulation.RemoteControl'
        if version != '':
            dispatch_string += f'.{version}'

        # Late-binding version
        #self.plantsim = win32.Dispatch(dispatch_string)

        # Early-binding version
        self.plantsim = win32.gencache.EnsureDispatch(dispatch_string)

        if visible:
            # Open the Plant Simulation window
            self.plantsim.SetVisible(True)

        if trust_models:
            # Allow Plant Simulation models to access the computer
            self.plantsim.SetTrustModels(True)

        self.license_type = license_type
        try:
            self.plantsim.SetLicenseType(self.license_type)
        except BaseException as e:
            if ErrorCode.extract(e.args) == -2147221503:
                raise Exception(f'The license type {self.license_type} does not seem to exist. Make sure it is a valid Plant Simluation license type.')

        self.load_model(model)
        self.set_path_context(path_context)
        self.set_event_controller()
        self.message_buffer = []
        if socket is not None:
            self.set_value(f"{socket}.Ein", True)

            self.plantsim_connection = PlantSimulationConnection()

    def load_model(self, filepath):

        print(f'Loading model "{filepath}"...\n')
        try:
            self.plantsim.LoadModel(filepath)
        except BaseException as e:
            if ErrorCode.extract(e.args) == -2147221503:
                raise Exception(f'The license server or the selected license type "{self.license_type}" is not available.\n'
                                + f'Make sure that the license server is up and running and you can connect to it (VPN etc.).\n'
                                + f'Make sure that a valid license of type "{self.license_type}" is available in the license server.')

    def set_event_controller(self):
        self.event_controller = f'{self.path_context}.Eventcontroller'

    def set_path_context(self, path_context):

        self.path_context = path_context
        self.plantsim.SetPathContext(self.path_context)

    def reset_simulation(self):

        if not self.event_controller:
            raise Exception('You need to set an event controller first!')
        self.plantsim.ResetSimulation(self.event_controller)
        self.message_buffer = []

    def start_simulation(self):

        if not self.event_controller:
            raise Exception('You need to set an event controller first!')

        self.plantsim.StartSimulation(self.event_controller)

    def stop_simulation(self):
        self.plantsim.StopSimulation()

    def get_object(self, object_name):
        # "Smart" getter that has some limited ability to decide which kind of object to return

        internal_class_name = self.get_value(f'{object_name}.internalClassName')

        if internal_class_name == 'AttributeExplorer':
            # Attribute explorer that dynamically fills a table
            return AttributeExplorer(self, object_name)

        elif internal_class_name == 'NwData':
            # Normal string
            return self.get_value(object_name)

        elif internal_class_name == 'NwList2D':
            # Normal string
            return Table(self, object_name)

        # Fallback: Return raw value of object
        else:
            return self.get_value(object_name)

    def get_value(self, object_name):

        return self.plantsim.GetValue(object_name)

    def set_value(self, object_name, value):

        self.plantsim.SetValue(object_name, value)

    def execute_simtalk(self, command_string, parameter=None, from_path_context=True):
        """
        Execute a SimTalk command accodring to COM documentation:
        PlantSim.ExecuteSimTalk("->real; return 3.14159")
        PlantSim.ExecuteSimTalk("param r:real->real; return r*r", 3.14159)
        :param command_string: Command to be executed
        :param parameter: (optional); parameter, if command contains a parameter to be set
        :param from_path_context: if true, command should be formulated form the path context.
                                Else, needs the whole path in the command
        """
        if from_path_context:
            command_string = f'{self.path_context}.{command_string}'
        else:
            command_string = f'.{command_string}'

        if parameter:
            self.plantsim.ExecuteSimTalk(command_string, parameter)
        else:
            self.plantsim.ExecuteSimTalk(command_string)

    def quit(self):

        self.plantsim.Quit()

    def get_current_state(self):
        state_data = []
        while len(state_data) == 0:
            state_data = self.get_object("CurrentState").rows_coldict.copy()
            #self.stop_simulation()
        return state_data[0]

    def get_next_message(self):
        if len(self.message_buffer) == 0:
            self.start_simulation()
            self.message_buffer = self.plantsim_connection.receive_messages()
            self.stop_simulation()
        return self.message_buffer.pop()

    def send_message(self, message):
        self.plantsim_connection.send_message(message)
