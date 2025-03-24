import yaml
import os
import sys
sys.path.append('.')
from croco_mpc_utils.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger


# # # # # # # # # # # # # # #
# EXPERIMENT LOADING UTILS  #
# # # # # # # # # # # # # # #

SUPPORTED_EXPERIMENTS = ['reach_ssqp',
                         'circle_ssqp', 
                         'circle_cssqp', 
                         'square_cssqp', 
                         'plane_cssqp',
                         'line_cssqp']


def is_valid_exp_name(EXP_NAME):
    '''
    Check that exp name is valid
    '''
    try: 
        assert(EXP_NAME in SUPPORTED_EXPERIMENTS)
    except NameError:
        logger.error("Error : config file name must be in "+str(SUPPORTED_EXPERIMENTS))
        
        
def load_config_file(EXP_NAME, path_prefix=''):
    '''
    Load YAML config file corresponding to an experiment name
    '''
    is_valid_exp_name(EXP_NAME)
    config_path = os.path.join(path_prefix, 'config/'+EXP_NAME+".yml")
    logger.debug("Opening config file "+str(config_path))
    with open(config_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def import_mpc_controller(EXP_NAME):
    '''
    Imports the MPC controller class corresponding to an experiment name
    '''
    is_valid_exp_name(EXP_NAME)
    if(EXP_NAME == 'reach_ssqp'):     
        from controllers.reach_ssqp   import KukaReachSSQP   as MPCController
    elif(EXP_NAME == 'circle_ssqp'):  
        from controllers.circle_ssqp  import KukaCircleSSQP  as MPCController
    elif(EXP_NAME == 'circle_cssqp'): 
        from controllers.circle_cssqp import KukaCircleCSSQP as MPCController
    elif(EXP_NAME == 'square_cssqp'): 
        from controllers.square_cssqp import KukaSquareCSSQP as MPCController
    elif(EXP_NAME == 'plane_cssqp'):  
        from controllers.plane_cssqp  import KukaPlaneCSSQP  as MPCController
    elif(EXP_NAME == 'line_cssqp'):  
        from controllers.line_cssqp  import KukaLineCSSQP  as MPCController
    logger.debug("Imported MPC controller for experiment : "+str(EXP_NAME))
    return MPCController


def get_log_config(EXP_NAME):
    '''
    Returns the log configuration for an experiment name
    '''
    is_valid_exp_name(EXP_NAME)
    if(EXP_NAME == 'reach_ssqp'):     
        log_config = SSQP_LOGS_REACH
    elif(EXP_NAME == 'circle_ssqp'):  
        log_config = SSQP_LOGS_CIRCLE
    elif(EXP_NAME == 'circle_cssqp'): 
        log_config = CSSQP_LOGS_CIRCLE
    elif(EXP_NAME == 'square_cssqp'): 
        log_config = CSSQP_LOGS_SQUARE
    elif(EXP_NAME == 'plane_cssqp'):  
        log_config = CSSQP_LOGS_PLANE
    elif(EXP_NAME == 'line_cssqp'):  
        log_config = CSSQP_LOGS_LINE
    logger.debug("Data log fields : "+str(log_config))
    return log_config



# # # # # # # # # # # # 
# DATA LOGGING UTILS  #
# # # # # # # # # # # # 

LOGS_NONE = []

SSQP_LOGS_MINIMAL = ['KKT', 
                     'ddp_iter',
                     't_child']

SSQP_LOGS_REACH = ['KKT', 
                   'ddp_iter',
                   't_child',
                   'joint_positions',
                   'x_des',
                   'tau',
                   'tau_ff',
                   'tau_gravity',
                   'cost',
                   'joint_torques_measured',
                   'joint_cmd_torques',
                   'target_position']

SSQP_LOGS_CIRCLE = ['KKT', 
                    'ddp_iter',
                    't_child',
                    'joint_positions',
                    'joint_velocities',
                    'x_des',
                    'tau',
                    'tau_ff',
                    'tau_gravity',
                    'cost',
                    'joint_torques_measured',
                    'joint_cmd_torques',
                    'target_position_x',
                    'target_position_y',
                    'target_position_z']

CSSQP_LOGS_MINIMAL = ['KKT', 
                     'ddp_iter',
                     't_child',
                     'qp_iters',
                     'cost',
                     'gap_norm',
                     'constraint_norm']

CSSQP_LOGS_CIRCLE = ['KKT', 
                     'ddp_iter',
                     't_child',
                     'qp_iters',
                     'cost',
                     'gap_norm',
                     'constraint_norm',
                     'joint_positions',
                     'joint_velocities',
                     'x_des',
                     'tau',
                     'tau_ff',
                     'tau_gravity',
                     'joint_torques_measured',
                     'joint_cmd_torques',
                     'target_position_x',
                     'target_position_y',
                     'target_position_z']

CSSQP_LOGS_SQUARE = ['KKT', 
                     'ddp_iter',
                     't_child',
                     'qp_iters',
                     'cost',
                     'gap_norm',
                     'constraint_norm',
                     'joint_positions',
                     'joint_velocities',
                     'x_des',
                     'tau',
                     'tau_ff',
                     'tau_gravity', 
                     'joint_torques_measured',
                     'joint_cmd_torques',
                     'target_position_x',
                     'target_position_y',
                     'target_position_z',
                     'lb',
                     'ub']

CSSQP_LOGS_PLANE = ['KKT', 
                    'ddp_iter',
                    't_child',
                    'qp_iters',
                    'cost',
                    'gap_norm',
                    'constraint_norm',
                    'joint_positions',
                    'joint_velocities',
                    'x_des',
                    'tau',
                    'tau_ff',
                    'tau_gravity',
                    'joint_torques_measured',
                    'joint_cmd_torques',
                    'ee_lb', 
                    'ee_ub',
                    'target_position']

CSSQP_LOGS_LINE = ['KKT', 
                   'ddp_iter',
                   't_child',
                   'qp_iters',
                   'cost',
                   'gap_norm',
                   'constraint_norm',
                   'joint_positions',
                   'joint_velocities',
                   'x_des',
                   'tau',
                   'tau_ff',
                   'tau_gravity',
                   'joint_torques_measured',
                   'joint_cmd_torques',
                   'lb', 
                   'ub',
                   'target_position_x',
                   'target_position_y',
                   'target_position_z']
