import yaml

SUPPORTED_EXPERIMENTS = ['reach_ssqp',
                         'circle_ssqp', 
                         'circle_cssqp', 
                         'square_cssqp', 
                         'plane_cssqp']

DGM_PARAMS_PATH = "/home/skleff/ws/workspace/install/robot_properties_kuka/lib/python3.8/site-packages/robot_properties_kuka/robot_properties_kuka/dynamic_graph_manager/dgm_parameters_iiwa.yaml"


def load_config_file_and_import_controller(EXP_NAME):
    # Check that exp name is valid
    try: 
        assert(EXP_NAME in SUPPORTED_EXPERIMENTS)
    except NameError:
        print("Error : config file name must be in "+str(SUPPORTED_EXPERIMENTS))
    # Load config file
    with open('config/'+EXP_NAME+".yml") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    # Import corresponding controller
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
    return data, MPCController

def get_log_config(EXP_NAME):
    if(EXP_NAME == 'reach_ssqp'):     
        return SSQP_LOGS_REACH
    elif(EXP_NAME == 'circle_ssqp'):  
        return SSQP_LOGS_CIRCLE
    elif(EXP_NAME == 'circle_cssqp'): 
        return CSSQP_LOGS_CIRCLE
    elif(EXP_NAME == 'square_cssqp'): 
        return CSSQP_LOGS_SQUARE
    elif(EXP_NAME == 'plane_cssqp'):  
        return CSSQP_LOGS_PLANE

LOGS_NONE = []

SSQP_LOGS_MINIMAL = ['KKT', 
                     'ddp_iter',
                     't_child']

SSQP_LOGS_REACH = ['KKT', 
                   'ddp_iter',
                   't_child',
                   'joint_positions',
                   'joint_torques',
                   'target_position_x',
                   'target_position_y',
                   'target_position_z']

SSQP_LOGS_CIRCLE = ['KKT', 
                    'ddp_iter',
                    't_child',
                    'joint_positions',
                    'joint_torques',
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
                     'joint_torques',
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
                     'joint_torques',
                     'target_position_x',
                     'target_position_y',
                     'target_position_z']

CSSQP_LOGS_PLANE = ['KKT', 
                    'ddp_iter',
                    't_child',
                    'qp_iters',
                    'cost',
                    'gap_norm',
                    'constraint_norm',
                    'joint_positions',
                    'joint_torques',
                    'target_position_x',
                    'target_position_y',
                    'target_position_z']