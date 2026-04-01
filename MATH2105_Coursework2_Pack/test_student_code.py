"""
Created in June 2021
Modified in March 2026

@authors: pmzejh, pmzmeh
"""

import numpy as np
import time
import re
import importlib
import matplotlib
import matplotlib.pyplot as plt
import errno
import sys



############################################################
case_definitions_dict = {
    ### Q1 ###
    'Q1 Case I': {
        'GeneralCommands': {
            'Import': 'numerical_calculus as n_cal',
            'Input1': 'x = 0.25; nvalues = 5',
            'Input2': 'h = 0.1 / (2**(np.arange(nvalues)))',
            'Input3': 'f = lambda x: np.sin(np.pi*x)',
            'Input4': 'd2fdx2 = lambda x: -np.pi*np.pi*np.sin(np.pi*x)',
            'Command1': {'cmd': 'n_cal.approximate_derivative(x,h,nvalues,f,d2fdx2,True)', 'Output1': 'derivative_approximations', 'Output2': 'derivative_errors', 'Output3': 'fig_q1'},
            },
        'Tests' : {
            'Test: approximations' : {
                'TestObject': 'derivative_approximations',
                'ObjectType': (np.ndarray)
                },
            'Test: errors' : {
                'TestObject': 'derivative_errors',
                'ObjectType': (np.ndarray)
                },
            'Test: plot' : {
                'TestObject': 'fig_q1',
                'ObjectType': (matplotlib.figure.Figure)
                }
            },
        },

    ### Q2 ###
    'Q2 Case I': {
        'GeneralCommands': {
            'Import': 'numerical_calculus as n_cal',
            'Input1': 'x = 0.25; r = 2.0; nlevels = 1; nvalues = 4',
            'Input2': 'h = 1.0 / (r**(np.arange(nvalues)))',
            'Input3': 'f = lambda x: np.sin(np.pi*x)',
            'Input4': 'd2fdx2 = lambda x: -np.pi*np.pi*np.sin(np.pi*x)',
            'Command1': {'cmd': 'n_cal.extrapolate_derivative(x,h,nvalues,nlevels,r,f,d2fdx2)', 'Output1': 'error_values', 'Output2': 'fig_q2'},
            },
        'Tests' : {
            'Test: errors' : {
                'TestObject': 'error_values',
                'ObjectType': (np.ndarray)
                },
            'Test: plot' : {
                'TestObject': 'fig_q2',
                'ObjectType': (matplotlib.figure.Figure)
                }
            }
        },

    ### Q3 ###
    'Q3 Case I': {
        'GeneralCommands': {
            'Import': 'numerical_calculus as n_cal',
            'Input1': 'a = 1.0; b = 2.0',
            'Input2': 'f = lambda x: np.sin(np.pi*x)',
            'Command1': {'cmd': 'n_cal.gauss_integration(a,b,f)', 'Output1': 'integral'},
            },
        'Tests' : {
            'Test: integral' : {
                'TestObject': 'integral',
                'ObjectType': (float)
                }
            }   
        },

    ### Q4 ###
    'Q4 Case I': {
        'GeneralCommands': {
            'Import': 'numerical_calculus as n_cal',
            'Input1': 'a = 1.0; b = 2.0; n = 10',
            'Input2': 'f = lambda x: np.sin(np.pi*x)',
            'Input3': 'f_int = lambda x: -np.cos(np.pi*x)/np.pi',
            'Command1': {'cmd': 'n_cal.composite_integration(a,b,n,f,f_int,n_cal.gauss_integration)', 'Output1': 'integral', 'Output2': 'error_value'}
            },
        'Tests' : {
            'Test: integral' : {
                'TestObject': 'integral',
                'ObjectType': (float)
                },
            'Test: error' : {
                'TestObject': 'error_value',
                'ObjectType': (float)
                }
            }
        },

    ### Q5 ###
    'Q5 Case I': {
        'GeneralCommands': {
            'Import': 'numerical_calculus as n_cal',
            'Input1': 'a = 1.0; b = 2.0',
            'Input2': 'npanels = 100 * 2**(np.arange(5))',
            'Input3': 'f = lambda x: np.sin(np.pi*x)',
            'Input4': 'd2fdx2 = lambda x: -np.pi*np.pi*np.sin(np.pi*x)',
            'Input5': 'd4fdx4 = lambda x: np.pi*np.pi*np.pi*np.pi*np.sin(np.pi*x)',
            'Input6': 'f_int = lambda x: -np.cos(np.pi*x)/np.pi',
            'Command1': {'cmd': 'n_cal.composite_errors(a,b,npanels,f,d2fdx2,d4fdx4,f_int)', 'Output1': 'error_values', 'Output2': 'error_bounds', 'Output3': 'fig_q5'},
                },
        'Tests' : {
            'Test: errors' : {
                'TestObject': 'error_values',
                'ObjectType': (np.ndarray)
                },
            'Test: bounds' : {
                'TestObject': 'error_bounds',
                'ObjectType': (np.ndarray)
                },
            'Test: plot' : {
                'TestObject': 'fig_q5',
                'ObjectType': (matplotlib.figure.Figure)
                }      
            }   
        },  

    ### Q6 ###
    'Q6 Case I': {
        'GeneralCommands': {
            'Import': 'numerical_calculus as n_cal',
            'Input1': 'a = 1.0; b = 2000.0',
            'Input2': 'Nmax = 200; TOL = 1e-10',
            'Input3': 'hinitial = 1000.0; vterm = 5.0',
            'Command1': {'cmd': 'n_cal.compute_time(a,b,Nmax,TOL,hinitial,vterm)', 'Output1': 'landing_time', 'Output2': 'niters'},
                },
        'Tests' : {
            'Test: time' : {
                'TestObject': 'landing_time',
                'ObjectType': (float)
                },
            'Test: iterations' : {
                'TestObject': 'niters',
                'ObjectType': (int)
                }
            }
        }

    }



############################################################
def run_test_case(dict_name,case_name,student_command_window):
    
    """
    Runs test case case_name from the dict_name
    It is assumed all files to import are in the current directory
    """

    # Change plt.show() - to avoid pauses when running the code
    plt.show2 = plt.show
    plt.show = lambda x=1: None
    
    student_command_window[case_name] = "<tt>"
    
    key_list = list(dict_name[case_name]["GeneralCommands"].keys())
    
    glob_dict = globals()
    loc_dict = {}


    #Import modules
    for key in key_list:
        if bool(re.search("^Import",key)):
            cmd = "import "+dict_name[case_name]["GeneralCommands"].get(key)

            student_command_window[case_name] = student_command_window[case_name]+cmd+"<br>\n"
            try:                
                exec(cmd,glob_dict,loc_dict)
            except Exception as e:
                student_command_window[case_name] = student_command_window[case_name]+f'<span style="color:red"> Error Raised: {e}</span><br>\n'

    #Variable set up (based on Inputs in dictionary)
    for key in key_list:
        if bool(re.search("^Input",key)):
            cmd = dict_name[case_name]["GeneralCommands"].get(key)
            student_command_window[case_name] = student_command_window[case_name]+cmd+"<br>\n"
            try:
                exec(cmd,glob_dict,loc_dict)
            except Exception as e:
                student_command_window[case_name] = student_command_window[case_name]+f'<span style="color:red"> Error Raised: {e}</span><br>\n'

    #Initialise the output dictionary
    dict_name[case_name]["Outputs"] = {}
    
    #Run the set of commands
    
    for key in key_list:
            if bool(re.search("^Command",key)):
                #Set up the outputs
                command_key_list = list(dict_name[case_name]["GeneralCommands"][key].keys())
                Outputs = ""
                for cmd_key in command_key_list:
                    if bool(re.search("^Output",cmd_key)):
                        Outputs = Outputs + dict_name[case_name]["GeneralCommands"][key].get(cmd_key) + ", "

                if len(Outputs) >= 3:
                    Outputs = Outputs[0:len(Outputs)-2]

                cmd = Outputs + " = " + dict_name[case_name]["GeneralCommands"][key].get("cmd")

                student_command_window[case_name] = student_command_window[case_name]+cmd+"<br>\n"
                try:
                    exec(cmd,glob_dict,loc_dict)
                        
                    #Append the outputs to the Outputs section of the case dictionary
                    for cmd_key in command_key_list:
                        if bool(re.search("^Output",cmd_key)):
                            output_name = dict_name[case_name]["GeneralCommands"][key].get(cmd_key)
                            dict_name[case_name]["Outputs"][output_name] = loc_dict.get(output_name)
                except Exception as e:
                    student_command_window[case_name] = student_command_window[case_name]+f'<span style="color:red"> Error Raised: {e}</span><br>\n'

    student_command_window[case_name] = student_command_window[case_name]+'</tt>'

    #Clean up all newly added modules
    for key in key_list:
        if bool(re.search("^Import",key)):
            if str.split(dict_name[case_name]["GeneralCommands"].get(key))[0] in sys.modules.keys(): 
                del sys.modules[str.split(dict_name[case_name]["GeneralCommands"].get(key))[0]] 
    
    # Change back plt.show()
    plt.show = plt.show2

    # Close all open figures
    plt.close('all')

############################################################

def create_html_of_outputs(student_case_dict,cmd_window):
    

    with open('StudentCodeTestOutput.html','w') as file:
        file.writelines('\n'.join(["<!DOCTYPE html>","<html>"]))
        file.write('<head> \n')
        file.write('Output from Code tests')
        file.write('</head> \n')

        file.write('<body> \n')

        case_keys = student_case_dict.keys()

        for case_name in case_keys:

            file.write('<p> <b>Case: '+case_name+'</b><br></p>\n')
            
            #Output the commands run
            file.write('<p style="margin-left:30px;"> <u>Commands Run:</u> </p>\n')
            file.write('<p style="margin-left:60px;"<tt>'+cmd_window[case_name]+'</tt></p>')
            
            key_list = list(student_case_dict[case_name]["Tests"].keys())

            for key in key_list:
                if bool(re.search("^Test",key)):
                    file.write('<pre><p style="margin-left:30px;"><u>'+key+'</u><br></p></pre>\n')
                    test_object_key = student_case_dict[case_name]["Tests"][key].get("TestObject")

                    if "Outputs" in student_case_dict[case_name]:
                            
                        student_output = student_case_dict[case_name]["Outputs"].get(test_object_key)

                        file.write('<p style="margin-left:60px;"> Student Output: </p>\n')
                        

                        student_output_type = type(student_output)
                        required_output_type = student_case_dict[case_name]["Tests"][key].get("ObjectType")
                        
                        if not isinstance(student_output,required_output_type):
                            required_string = str(required_output_type).replace("<","&lt")
                            required_string = required_string.replace(">","&gt")
                            received_string = str(student_output_type).replace("<","&lt")
                            received_string = received_string.replace(">","&gt")
                            warn = "requires (one of) <tt>"+required_string+"</tt> received <tt>"+received_string+"</tt>"
                            file.write("<p style=\"margin-left:60px;\"><span style=\"color:red\">Warning</span>: Student output is of incorrect type, "+warn+"</p>\n")

                        file.write('<pre><p style="margin-left:60px;">'+ test_object_key + ' = </p></pre>')
                            
                        if isinstance(student_output,matplotlib.figure.Figure):
                            student_output.savefig(test_object_key+".png",bbox_inches = "tight")
                                
                            file.write('<p style="margin-left:90px;"><img src="'+test_object_key+".png\" height=\"400\" width=\"600\"></p><br><br>\n")
                        else:
                            student_output = str(student_output)
                            
                            #if isinstance(student_output,str):
                            student_output = student_output.replace("<","&lt")
                            student_output = student_output.replace(">","&gt")
                            student_output = student_output.replace('\n','<br>')
                            #student_output = '<pre> '+student_output+' </pre>'
                            
                            file.write('<pre><p style="margin-left:90px;">'+ student_output +'</p></pre>')
                    else:
                        file.write('<p style="margin-left:60px;"> Student Output: None </p>\n')

                
                        
        file.write('</body> \n')
        file.write('</html> \n')


############################################################
#Test the code and output

case_keys = case_definitions_dict.keys()

student_command_window = {}

for case_name in case_keys:

    print("      Running ",case_name)
    #Run the student code and store output
    run_test_case(case_definitions_dict,case_name,student_command_window)

create_html_of_outputs(case_definitions_dict,student_command_window)
print("      Created file StudentCodeTestOutput.html")
print("          (open it in a web browser)")


