from symbolic_pp.DeepPoly_quadapter import *
from utils.quadapter_utils import *

import math
from gurobipy import GRB
import gurobipy as gp
import time
import numpy as np


class LayerEncoding:
    def __init__(
            self,
            gp_model,
            preimg_mode,
            layer_index,
            layer_size,
            layer_paras,
            bit_lb,
            bit_ub,
            if_hid,
    ):
        self.layer_index = layer_index
        self.layer_size = layer_size
        self.layer_paras = layer_paras  # weight+bias
        self.bit_lb = bit_lb
        self.bit_ub = bit_ub
        self.frac_bit = None
        self.grad = None
        self.realVal = None

        if if_hid:
            neuron_lb_after = 0
        else:
            neuron_lb_after = -GRB.MAXINT

        neuron_lb_before = -GRB.MAXINT

        self.lb = np.zeros(layer_size, dtype=np.float32)
        self.ub = np.zeros(layer_size, dtype=np.float32)
        self.clipped_lb = np.zeros(layer_size, dtype=np.float32)
        self.clipped_ub = np.zeros(layer_size, dtype=np.float32)

        self.qu_lb = np.zeros(layer_size, dtype=np.float32)  # lower bound of QNN
        self.qu_ub = np.zeros(layer_size, dtype=np.float32)  # upper bound of QNN
        self.qu_clipped_lb = np.zeros(layer_size, dtype=np.float32)
        self.qu_clipped_ub = np.zeros(layer_size, dtype=np.float32)

        # variable set for encoding bit-width of current layer
        self.bit_vars = [gp_model.addVar(vtype=GRB.BINARY) for i in range(self.bit_ub - self.bit_lb + 1)]

        # obtain minimal bit-width for integer part to avoid overflow
        if layer_index > 0:
            self.max_weight = np.round(max(np.max(layer_paras[0]), np.max(layer_paras[1])))
            self.min_weight = np.round(min(np.min(layer_paras[0]), np.min(layer_paras[1])))
            self.max_int = max(abs(self.max_weight), abs(self.min_weight))
            if self.max_int == 0:
                self.int_bit = 1
            elif self.max_int == 1:
                self.int_bit = 2
            else:
                self.int_bit = int(np.ceil(math.log(self.max_int, 2)) + 1)
        else:
            self.int_bit = None

        self.relaxed_lb = np.zeros(layer_size, dtype=np.float32)
        self.relaxed_lb_expression = [1 for i in range(layer_size)]
        self.relaxed_ub = np.zeros(layer_size, dtype=np.float32)
        self.relaxed_ub_expression = [1 for i in range(layer_size)]
        self.actMode = np.zeros(layer_size, dtype=np.float32)

        #### initiate variables for encoding preimage template


        if preimg_mode == 'milp' or preimg_mode == 'comp':
            self.gp_vars_before = [gp_model.addVar(lb=neuron_lb_before, ub=1000, vtype=GRB.CONTINUOUS) for s in
                                      range(layer_size)]  # before relu function
            self.gp_vars_after = [gp_model.addVar(lb=0, ub=1000, vtype=GRB.CONTINUOUS) for s in
                                      range(layer_size)]  # before relu function
            self.alpha = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]
            self.beta = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]

        elif preimg_mode == 'abstr' or preimg_mode == 'comp':
            self.gp_vars_lb_before = [gp_model.addVar(lb=neuron_lb_before, ub=1000, vtype=GRB.CONTINUOUS) for s in
                                      range(layer_size)]  # before relu function
            self.gp_vars_ub_before = [gp_model.addVar(lb=neuron_lb_before, ub=1000, vtype=GRB.CONTINUOUS) for s in
                                      range(layer_size)]  # before relu function
            self.alpha_before = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]
            self.alpha_after = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]
            self.beta_before = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]
            self.beta_after = [gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS) for s in range(layer_size)]
        else:
            print("Wrong option for the preimage computation mode!")
            exit(0)

        gp_model.update()

        print("The quantization bit size for integer parts of Layer ", self.layer_index, " is: ", self.int_bit)

    def set_input_bounds(self, low, high):
        self.lb = low
        self.ub = high

    def set_realVal(self, realVal):
        self.realVal = realVal
        print("We set the real output values for the layer ", self.layer_index)


class GPEncoding:
    def __init__(self, arch, model, args, original_prediction, x_low_real, x_high_real):
        self.gp_model = gp.Model("gp_encoding")
        self.tole = 1e-6
        self.gp_model.Params.IntFeasTol = 1e-9
        self.gp_model.Params.FeasibilityTol = self.tole
        self.gp_model.setParam(GRB.Param.Threads, 30)
        self.gp_model.setParam(GRB.Param.OutputFlag, 0)
        self.bit_lb = args.bit_lb
        self.bit_ub = args.bit_ub
        self.preimg_mode = args.preimg_mode  # preimage computation mode
        self.x_low_real = x_low_real  # lower bound of input region
        self.x_high_real = x_high_real  # upper bound of input region
        self.sample_id = args.sample_id
        self.eps = args.eps  # perturbation radius
        self.outputPath = args.outputPath
        self.ifRelax = args.ifRelax
        self.scaleValueSet = []

        self._stats = {
            "encoding_time": 0,
            "solving_time": 0,
            "backward_time": 0,
            "forward_time": 0,
            "total_time": 0,
        }

        self.dense_layers = []
        self.nnparas = []
        self.deep_model = model
        self.layerNum = len(model.dense_layers)
        self.targetCls = original_prediction
        self.deepPolyNets_DNN = DP_DNN_network(True)

        self.input_gp_vars = []
        for i, l in enumerate(model.dense_layers):
            tf_layer = model.dense_layers[i]
            w_cont, b_cont = tf_layer.get_weights()
            paras = [w_cont.T, b_cont]
            self.nnparas.append(paras)

        ########## output layer
        self.output_layer = LayerEncoding(self.gp_model, preimg_mode=self.preimg_mode,
                                          layer_index=len(self.nnparas),
                                          layer_size=arch[-1],
                                          layer_paras=self.nnparas[-1], bit_lb=self.bit_lb, bit_ub=self.bit_ub,
                                          if_hid=False)

        ########## hidden layer
        for layer in range(len(arch) - 2):
            self.dense_layers.append(
                LayerEncoding(self.gp_model, preimg_mode=self.preimg_mode,
                              layer_index=layer + 1,
                              layer_size=arch[layer + 1],
                              layer_paras=self.nnparas[layer],
                              bit_lb=self.bit_lb, bit_ub=self.bit_ub,
                              if_hid=True)
            )
            self.scaleValueSet.append(0)

        ########## input layer
        input_size = arch[0]

        self.input_layer = LayerEncoding(self.gp_model, preimg_mode=self.preimg_mode,
                                         layer_index=0,
                                         layer_size=input_size,
                                         layer_paras=None,
                                         bit_lb=self.bit_lb, bit_ub=self.bit_ub,
                                         if_hid=False)

        self.deepPolyNets_DNN.load_dnn(model)

        # add input vars constraints
        for input_index in range(self.input_layer.layer_size):
            x_lb = x_low_real[input_index]
            x_ub = x_high_real[input_index]
            cur_var = self.gp_model.addVar(lb=x_lb, ub=x_ub, vtype=GRB.CONTINUOUS)
            self.input_gp_vars.append(cur_var)

    def verified_quant(self, lb, ub):

        self.assert_input_box(lb, ub)

        self.symbolic_propagate()

        # DNN should satisfy the property
        out_bounds_lb = self.output_layer.lb
        out_bounds_ub = self.output_layer.ub
        other_max = -1000

        for i, v in enumerate(self.output_layer.ub):
            if i == self.targetCls:
                continue
            else:
                other_max = max(other_max, v)

        print("The lower bound of the target class: ", out_bounds_lb[self.targetCls])
        print("The maximal upper bound of other classes: ", other_max)

        if (out_bounds_lb[self.targetCls] >= other_max):
            backward_start_time = time.time()
            self.backward_preimage_computation()
            backward_end_time = time.time()
            print("Backward Time is: ", backward_end_time - backward_start_time)

            ifSucc, qu_list, qu_frac_list, qu_int_list = self.forward_quantization()
            forward_end_time = time.time()
            print("Forward time is: ", forward_end_time - backward_end_time)

            self._stats["backward_time"] = backward_end_time - backward_start_time
            self._stats["forward_time"] = forward_end_time - backward_end_time
            self._stats["total_time"] = self._stats["backward_time"] + self._stats["forward_time"]

            return ifSucc, qu_list, qu_frac_list, qu_int_list

        else:
            print("The property does not hold in DNN!")
            exit(0)

    # initiate input region
    def assert_input_box(self, x_lb, x_ub):
        low, high = x_lb, x_ub

        input_size = self.input_layer.layer_size

        # Ensure low is a vector
        low = np.array(low, dtype=np.float32) * np.ones(input_size, dtype=np.float32)
        high = np.array(high, dtype=np.float32) * np.ones(input_size, dtype=np.float32)

        self.input_layer.set_input_bounds(low, high)

        self.deepPolyNets_DNN.property_region = 1

        for i in range(self.deepPolyNets_DNN.layerSizes[0]):
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_lower = low[i]
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_upper = high[i]
            self.deepPolyNets_DNN.property_region *= (high[i] - low[i])
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_lower = np.array([low[i]])
            self.deepPolyNets_DNN.layers[0].neurons[i].concrete_algebra_upper = np.array([high[i]])
            self.deepPolyNets_DNN.layers[0].neurons[i].algebra_lower = np.array([low[i]])
            self.deepPolyNets_DNN.layers[0].neurons[i].algebra_upper = np.array([high[i]])

    # Conduct DeepPoly on DNN
    def symbolic_propagate(self):
        self.deepPolyNets_DNN.deeppoly()

        for i, l in enumerate(self.dense_layers):

            for out_index in range(l.layer_size):
                lb = self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[out_index].concrete_lower_noClip
                ub = self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[out_index].concrete_upper_noClip

                lb_clipped = self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[
                    out_index].concrete_lower
                ub_clipped = self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[
                    out_index].concrete_upper

                l.lb[out_index] = lb
                l.ub[out_index] = ub

                l.clipped_lb[out_index] = max(lb_clipped, 0)
                l.clipped_ub[out_index] = max(ub_clipped, 0)

                # record activation pattern for activation-based preimage computation method
                if self.preimg_mode == 'abstr' or self.preimg_mode == 'comp':
                    act_mode = self.deepPolyNets_DNN.layers[2 * (i + 1)].neurons[out_index].actMode
                    l.actMode[out_index] = act_mode

        for out_index in range(self.output_layer.layer_size):
            lb = self.deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_lower_noClip
            ub = self.deepPolyNets_DNN.layers[-1].neurons[out_index].concrete_upper_noClip
            self.output_layer.lb[out_index] = lb
            self.output_layer.ub[out_index] = ub

    # TODO: Design a composition method (Currently Not in use)
    # if abstraction-based method compute a relatively tight preimage, we turn to use MILP-based method
    def backward_preimage_computation_composed(self):
        cur_layer = self.output_layer
        in_layer_index = len(self.dense_layers)

        metric = 0.1
        # add input_layer's region constraint

        for in_layer in reversed(self.dense_layers):

            in_layer_index -= 1
            scaleValue = self.underPreImageMILP(in_layer_index, in_layer, cur_layer)

            if scaleValue == 0:  # abstraction of current layer is better, hence use abstr-based method
                scaleValue = self.underPreImageAbstr(in_layer_index, in_layer, cur_layer)

            self.scaleValueSet[in_layer.layer_index - 1] = scaleValue
            cur_layer = in_layer


    # Two backward preimage computation: MILP-based, abstraction-bassed
    def backward_preimage_computation(self):
        cur_layer = self.output_layer
        in_layer_index = len(self.dense_layers)

        for in_layer in reversed(self.dense_layers):

            in_layer_index -= 1
            scaleValue = 0

            if self.preimg_mode == 'milp' or self.preimg_mode == 'comp':
                scaleValue = self.underPreImageMILP(in_layer_index, in_layer, cur_layer)

            if self.preimg_mode == 'abstr' or (self.preimg_mode == 'comp' and scaleValue <= 0):
                scaleValue = self.underPreImageAbstr(in_layer_index, in_layer, cur_layer)

            self.scaleValueSet[in_layer.layer_index - 1] = scaleValue
            cur_layer = in_layer

    # MILP-based method for computing preimage
    def underPreImageMILP(self, in_layer_index, in_layer, cur_layer):
        enc_start_time = time.time()

        var_ll = []
        prop_cstr_ll = []
        model_cstr_ll = []
        w = cur_layer.layer_paras[0]
        b = cur_layer.layer_paras[1]

        relaxScale = self.gp_model.addVar(lb=0, ub=100, vtype=GRB.CONTINUOUS)

        relaxScale_LL = []
        relaxScale_LL.append(relaxScale)

        for in_index in range(in_layer.layer_size):
            neuron_val = in_layer.realVal[in_index]
            neuron_lb = in_layer.lb[in_index]
            neuron_ub = in_layer.ub[in_index]

            alpha_K = max(neuron_val - neuron_lb, 1e-3)
            beta_K = max(neuron_ub - neuron_val, 1e-3)

            # alpha_K = (neuron_ub - neuron_lb)/2
            # beta_K = (neuron_ub - neuron_lb)/2

            model_cstr_ll.append(
                self.gp_model.addConstr(in_layer.alpha[in_index] == (alpha_K * relaxScale)))
            model_cstr_ll.append(
                self.gp_model.addConstr(in_layer.beta[in_index] == (beta_K * relaxScale)))

            model_cstr_ll.append(
                self.gp_model.addConstr(
                    in_layer.ub[in_index] + beta_K * relaxScale >= in_layer.lb[in_index] - alpha_K * relaxScale))

            # get symbolic lower bounds w.r.t. input vars
            in_lb_algebra = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                in_index].concrete_algebra_lower
            in_ub_algebra = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                in_index].concrete_algebra_upper

            relaxed_lb_bias = in_lb_algebra[-1] - in_layer.alpha[in_index]
            relaxed_ub_bias = in_ub_algebra[-1] + in_layer.beta[in_index]

            # get symbolic upper bounds w.r.t. input vars
            symbolic_lb_expression = np.dot(in_lb_algebra[:-1], self.input_gp_vars)
            symbolic_lb_expression = symbolic_lb_expression + relaxed_lb_bias

            symbolic_ub_expression = np.dot(in_ub_algebra[:-1], self.input_gp_vars)
            symbolic_ub_expression = symbolic_ub_expression + relaxed_ub_bias

            model_cstr_ll.append(
                self.gp_model.addConstr(in_layer.gp_vars_before[in_index] <= symbolic_ub_expression))
            model_cstr_ll.append(
                self.gp_model.addConstr(in_layer.gp_vars_before[in_index] >= symbolic_lb_expression))

            model_cstr_ll.append(
                self.gp_model.addGenConstrMax(in_layer.gp_vars_after[in_index],
                                              [in_layer.gp_vars_before[in_index], 0]))

        self.gp_model.update()

        # encoding cur_layer's computation
        for out_index in range(cur_layer.layer_size):
            weights = w[out_index]
            bias = b[out_index]

            accumulation = np.dot(weights, in_layer.gp_vars_after)
            accumulation = accumulation + bias

            model_cstr_ll.append(self.gp_model.addConstr(cur_layer.gp_vars_before[out_index] == accumulation))

        enc_finish_time = time.time()

        prop_start_time = time.time()

        # encoding cur_layer's property, compute concrete bounds
        if cur_layer.layer_index == (len(self.dense_layers) + 1):
            # output layer: not argmax
            other_vars = []
            for i in range(cur_layer.layer_size):
                if i == int(self.targetCls):
                    continue
                else:
                    other_vars.append(cur_layer.gp_vars_before[i])

            other_maximal = self.gp_model.addVar(lb=-1000, vtype=GRB.CONTINUOUS)
            prop_cstr_ll.append(self.gp_model.addGenConstrMax(other_maximal, other_vars))
            prop_cstr_ll.append(
                self.gp_model.addConstr(other_maximal >= cur_layer.gp_vars_before[self.targetCls] + self.tole))

        else:
            # other hidden layer: exist one neuron s.t. after relu, it is not included in the clipped_relaxed_ub
            bigM = 1000
            sumOfK = 0
            for i in range(cur_layer.layer_size):
                k_i_lb = self.gp_model.addVar(vtype=GRB.BINARY)
                relaxScale_LL.append(k_i_lb)

                prop_cstr_ll.append(self.gp_model.addConstr(
                    cur_layer.gp_vars_before[i] <= cur_layer.relaxed_lb_expression[i] - bigM * (
                            k_i_lb - 1) - 2 * self.tole))
                prop_cstr_ll.append(self.gp_model.addConstr(
                    cur_layer.gp_vars_before[i] >= cur_layer.relaxed_lb_expression[
                        i] - bigM * k_i_lb + 2 * self.tole))
                sumOfK = sumOfK + k_i_lb
                #
                # k_i encodes: is not included
                # for upper bounds
                k_i_ub = self.gp_model.addVar(vtype=GRB.BINARY)
                relaxScale_LL.append(k_i_ub)
                prop_cstr_ll.append(self.gp_model.addConstr(
                    cur_layer.gp_vars_before[i] >= cur_layer.relaxed_ub_expression[i] + bigM * (
                            k_i_ub - 1) + 2 * self.tole))
                prop_cstr_ll.append(self.gp_model.addConstr(
                    cur_layer.gp_vars_before[i] <= cur_layer.relaxed_ub_expression[
                        i] + bigM * k_i_ub - 2 * self.tole))

                sumOfK = sumOfK + k_i_ub

            prop_cstr_ll.append(self.gp_model.addConstr(sumOfK >= 1))

        prop_finish_time = time.time()
        prop_encoding_time = prop_finish_time - prop_start_time

        self.gp_model.update()
        self.gp_model.setObjective(relaxScale, GRB.MINIMIZE)
        self.gp_model.update()
        self.gp_model.setParam('DualReductions', 0)  # set this value to 0, to get a more definite result
        opt_start_time = time.time()

        self.gp_model.optimize()

        opt_finish_time = time.time()
        optimization_time = opt_finish_time - opt_start_time

        ifgpINF_OR_UNBD = self.gp_model.status == GRB.INF_OR_UNBD
        ifgpINFEASIBLE = self.gp_model.status == GRB.INFEASIBLE
        ifgpUNBOUNDED = self.gp_model.status == GRB.UNBOUNDED
        ifgpOptimize = self.gp_model.status == GRB.OPTIMAL

        # print("ifgpINF_OR_UNBD: ", ifgpINF_OR_UNBD)
        # print("ifgpINFEASIBLE: ", ifgpINFEASIBLE)
        # print("ifgpOptimize: ", ifgpOptimize)
        # print("ifgpUNBOUNDED: ", ifgpUNBOUNDED)

        scaleValue = -10000

        if ifgpOptimize:
            scaleValue = relaxScale.X
            print("\n########################### scaleValue for Relax layer index using MILP-based method: ",
                  in_layer_index, " is: ",
                  scaleValue, " ###########################")

            for in_index in range(in_layer.layer_size):
                alpha = in_layer.alpha[in_index].X
                beta = in_layer.beta[in_index].X

                relaxed_ub = in_layer.ub[in_index] + beta
                relaxed_lb = in_layer.lb[in_index] - alpha
                in_layer.relaxed_ub[in_index] = relaxed_ub
                in_layer.relaxed_lb[in_index] = relaxed_lb

                in_lb_algebra = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                    in_index].concrete_algebra_lower
                in_ub_algebra = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                    in_index].concrete_algebra_upper

                relaxed_lb_bias = in_lb_algebra[-1] - alpha
                relaxed_ub_bias = in_ub_algebra[-1] + beta

                # get symbolic upper bounds w.r.t. input vars
                relaxed_symbolic_lb_expression = np.dot(in_lb_algebra[:-1], self.input_gp_vars)
                relaxed_symbolic_lb_expression = relaxed_symbolic_lb_expression + relaxed_lb_bias

                relaxed_symbolic_ub_expression = np.dot(in_ub_algebra[:-1], self.input_gp_vars)
                relaxed_symbolic_ub_expression = relaxed_symbolic_ub_expression + relaxed_ub_bias

                in_layer.relaxed_lb_expression[in_index] = relaxed_symbolic_lb_expression
                in_layer.relaxed_ub_expression[in_index] = relaxed_symbolic_ub_expression

                if relaxed_ub <= 0:
                    in_layer.relaxed_ub_expression[in_index] = 0

            self.gp_model.remove(prop_cstr_ll)
            self.gp_model.remove(model_cstr_ll)
            self.gp_model.remove(relaxScale_LL)
            self.gp_model.remove(var_ll)
            self.gp_model.update()

        return scaleValue

    def underPreImageAbstr(self, in_layer_index, in_layer, cur_layer):

        enc_start_time = time.time()
        relaxScale_LL = []

        var_ll = []
        prop_cstr_ll = []
        model_cstr_ll = []
        w = cur_layer.layer_paras[0]
        b = cur_layer.layer_paras[1]

        relaxScale = self.gp_model.addVar(lb=0, ub=1000, vtype=GRB.CONTINUOUS)
        relaxScale_LL.append(relaxScale)

        # define relaxed_region for gp_vars_after (ReLU is approximated by abstract transformers)
        # We require that activation pattern remains unchanged in the preimage to avoid combinatorial explosion problem
        for in_index in range(in_layer.layer_size):

            neuron_val = in_layer.realVal[in_index]
            actMode = in_layer.actMode[in_index]

            neuron_lb = in_layer.lb[in_index]
            neuron_ub = in_layer.ub[in_index]

            if actMode == 1:
                alpha_K = neuron_val - neuron_lb
                beta_K = neuron_ub - neuron_val

                model_cstr_ll.append(
                    self.gp_model.addConstr(in_layer.alpha_before[in_index] == (alpha_K * relaxScale)))
                model_cstr_ll.append(
                    self.gp_model.addConstr(in_layer.beta_after[in_index] == (beta_K * relaxScale)))

                model_cstr_ll.append(self.gp_model.addGenConstrMin(in_layer.alpha_after[in_index],
                                                                   [in_layer.alpha_before[in_index],
                                                                    in_layer.lb[in_index]]))
            elif actMode == 2:
                continue
            else:  # actMode == 3 or 4:
                model_cstr_ll.append(
                    self.gp_model.addConstr(in_layer.alpha_after[in_index] == (-neuron_lb * relaxScale)))
                model_cstr_ll.append(
                    self.gp_model.addConstr(in_layer.beta_after[in_index] == (neuron_ub * relaxScale)))

        self.gp_model.update()

        # compute relaxed accumulated bounds instead of exactly encoding cur_layer's computation
        for out_index in range(cur_layer.layer_size):
            weights = w[out_index]
            tmp_add_lower = 0
            tmp_add_upper = 0

            # get new added biases
            for in_index in range(in_layer.layer_size):
                actMode = in_layer.actMode[in_index]
                if actMode == 1:
                    if weights[in_index] >= 0:
                        tmp_add_lower -= weights[in_index] * in_layer.alpha_after[in_index]
                        tmp_add_upper += weights[in_index] * in_layer.beta_after[in_index]
                    else:
                        tmp_add_lower += weights[in_index] * in_layer.beta_after[in_index]
                        tmp_add_upper -= weights[in_index] * in_layer.alpha_after[in_index]
                elif actMode == 2:
                    continue

                elif actMode == 3:
                    # update bounds
                    K = in_layer.ub[in_index] / (in_layer.ub[in_index] - in_layer.lb[in_index])
                    if weights[in_index] >= 0:
                        tmp_add_upper += weights[in_index] * K * (
                                in_layer.beta_after[in_index] + in_layer.alpha_after[in_index])
                    else:
                        tmp_add_lower += weights[in_index] * K * (
                                in_layer.beta_after[in_index] + in_layer.alpha_after[in_index])

                else:  # actMode == 4
                    K = in_layer.ub[in_index] / (in_layer.ub[in_index] - in_layer.lb[in_index])
                    if weights[in_index] >= 0:
                        tmp_add_lower -= weights[in_index] * in_layer.alpha_after[in_index]
                        tmp_add_upper += weights[in_index] * K * (
                                in_layer.beta_after[in_index] + in_layer.alpha_after[in_index])
                    else:
                        tmp_add_lower += weights[in_index] * K * (
                                in_layer.beta_after[in_index] + in_layer.alpha_after[in_index])
                        tmp_add_upper -= weights[in_index] * in_layer.alpha_after[in_index]

            model_cstr_ll.append(self.gp_model.addConstr(
                (tmp_add_lower + cur_layer.lb[out_index]) == cur_layer.gp_vars_lb_before[out_index]))
            model_cstr_ll.append(self.gp_model.addConstr(
                (tmp_add_upper + cur_layer.ub[out_index]) == cur_layer.gp_vars_ub_before[out_index]))

            self.gp_model.update()

        enc_finish_time = time.time()
        model_encoding_time = enc_finish_time - enc_start_time

        prop_start_time = time.time()

        # encoding cur_layer's property, the preimage propagated should be concluded in the next layer's preimage computed before
        if cur_layer.layer_index == (len(self.dense_layers) + 1):
            for var_index, var in enumerate(cur_layer.gp_vars_ub_before):
                if var_index == self.targetCls:
                    continue
                elif var_index < self.targetCls:
                    prop_cstr_ll.append(self.gp_model.addConstr(
                        cur_layer.gp_vars_lb_before[self.targetCls] >= (var + 2 * self.tole)))
                else:
                    prop_cstr_ll.append(self.gp_model.addConstr(cur_layer.gp_vars_lb_before[self.targetCls] >= var))
        else:
            for var_index, var in enumerate(cur_layer.gp_vars_lb_before):
                if cur_layer.actMode[var_index] == 1:
                    prop_cstr_ll.append(self.gp_model.addConstr(
                        cur_layer.gp_vars_ub_before[var_index] <= cur_layer.relaxed_ub[var_index]))
                    prop_cstr_ll.append(self.gp_model.addConstr(
                        cur_layer.gp_vars_lb_before[var_index] >= cur_layer.relaxed_lb[var_index]))
                elif cur_layer.actMode[var_index] == 2:
                    prop_cstr_ll.append(self.gp_model.addConstr(
                        cur_layer.gp_vars_ub_before[var_index] <= 0))  # relaxed_ub>=0
                else:
                    prop_cstr_ll.append(self.gp_model.addConstr(
                        cur_layer.gp_vars_ub_before[var_index] <= cur_layer.relaxed_ub[var_index]))  # relaxed_ub>=0
                    prop_cstr_ll.append(self.gp_model.addConstr(
                        cur_layer.gp_vars_lb_before[var_index] >= cur_layer.relaxed_lb[var_index]))

        self.gp_model.update()

        self.gp_model.setObjective(relaxScale, GRB.MAXIMIZE)
        self.gp_model.update()
        self.gp_model.setParam('DualReductions', 0)  # set this value to 0, to get a more definite result
        opt_start_time = time.time()

        self.gp_model.optimize()

        opt_finish_time = time.time()
        optimization_time = opt_finish_time - opt_start_time

        ifgpINF_OR_UNBD = self.gp_model.status == GRB.INF_OR_UNBD
        ifgpINFEASIBLE = self.gp_model.status == GRB.INFEASIBLE
        ifgpUNBOUNDED = self.gp_model.status == GRB.UNBOUNDED
        ifgpOptimize = self.gp_model.status == GRB.OPTIMAL

        # print("ifgpINF_OR_UNBD: ", ifgpINF_OR_UNBD)
        # print("ifgpINFEASIBLE: ", ifgpINFEASIBLE)
        # print("ifgpOptimize: ", ifgpOptimize)
        # print("ifgpUNBOUNDED: ", ifgpUNBOUNDED)

        if ifgpOptimize:
            scaleValue = relaxScale.X
            print("\n########################### scaleValue for Relax layer index using abstraction-based method: ",
                  in_layer_index, " is: ",
                  scaleValue, " ###########################")

            for in_index in range(in_layer.layer_size):

                alpha_after = in_layer.alpha_after[in_index].X
                beta_after = in_layer.beta_after[in_index].X

                if in_layer.ub[in_index] <= 0:  # Case B
                    in_layer.relaxed_ub[in_index] = 0
                    in_layer.relaxed_lb[in_index] = in_layer.lb[in_index] - alpha_after

                else:  # Case A,C,D
                    in_layer.relaxed_ub[in_index] = np.float32(in_layer.ub[in_index] + beta_after)
                    in_layer.relaxed_lb[in_index] = np.float32(in_layer.lb[in_index] - alpha_after)

                in_lb_algebra = deepcopy(self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                                             in_index].concrete_algebra_lower)
                in_ub_algebra = deepcopy(self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[
                                             in_index].concrete_algebra_upper)

                relaxed_lb_bias = in_lb_algebra[-1] - alpha_after
                relaxed_ub_bias = in_ub_algebra[-1] + beta_after

                relaxed_symbolic_lb_expression = np.dot(in_lb_algebra[:-1], self.input_gp_vars)
                relaxed_symbolic_lb_expression = relaxed_symbolic_lb_expression + relaxed_lb_bias
                relaxed_symbolic_ub_expression = np.dot(in_ub_algebra[:-1], self.input_gp_vars)
                relaxed_symbolic_ub_expression = relaxed_symbolic_ub_expression + relaxed_ub_bias

                in_layer.relaxed_lb_expression[in_index] = relaxed_symbolic_lb_expression
                in_layer.relaxed_ub_expression[in_index] = relaxed_symbolic_ub_expression

                if in_layer.ub[in_index] <= 0:
                    in_layer.relaxed_ub_expression[in_index] = 0

            self.gp_model.remove(prop_cstr_ll)
            self.gp_model.remove(model_cstr_ll)
            self.gp_model.remove(relaxScale_LL)
            self.gp_model.remove(var_ll)
            self.gp_model.update()

            return scaleValue

    # forward quantization procedure
    def forward_quantization(self):
        print("\nNow we begin to do the forward quantization!")
        qu_list = []
        qu_frac_list = []
        qu_int_list = []

        nonInputLayers = self.dense_layers.copy()
        nonInputLayers.append(self.output_layer)

        in_layer_index = -1

        for cur_layer in nonInputLayers:
            in_layer_index += 1

            if cur_layer.layer_index == 1:
                in_layer = self.input_layer
            else:
                in_layer = self.dense_layers[cur_layer.layer_index - 2]

            w = cur_layer.layer_paras[0]
            b = cur_layer.layer_paras[1]

            # test for all bit:
            lower_bit = self.bit_lb
            upper_bit = self.bit_ub

            ifFound = False

            for rela_bit in range(upper_bit - lower_bit + 1):
                pre_mul_qu_lb_deepPoly = []
                pre_mul_qu_ub_deepPoly = []

                if ifFound:
                    break

                model_cstr_ll = []
                prop_cstr_ll = []
                var_ll = []

                frac_bit = rela_bit + lower_bit
                int_bit = cur_layer.int_bit
                all_bit = frac_bit + int_bit

                # get_quantized_paras
                qu_w = quantize_int(w, all_bit, frac_bit) / (2 ** frac_bit)
                qu_b = quantize_int(b, all_bit, frac_bit) / (2 ** frac_bit)

                # for last layer
                target_lb = 0
                other_ubs = []

                # quantized_concrete_algebra_lower
                quantized_concrete_algebra_lower = []
                quantized_concrete_algebra_upper = []

                sumOfK = 0
                numOfK = 0

                for out_index in range(cur_layer.layer_size):
                    qu_weights = qu_w[out_index]
                    qu_bias = qu_b[out_index]

                    tmp_acc_lower = 0
                    tmp_acc_upper = 0

                    # for var_index_poly in range(self.bit_ub - self.bit_lb + 1):
                    lower_bound = np.append(qu_weights, qu_bias)  # cur_layer's paras (size of input_layer)
                    upper_bound = np.append(qu_weights, qu_bias)  # cur_layer's paras (size of input_layer)

                    # reverse, from cur_layer's affine layer to input layer
                    cur_neuron_concrete_algebra_lower = None
                    cur_neuron_concrete_algebra_upper = None

                    if in_layer_index == 0:
                        cur_neuron_concrete_algebra_lower = deepcopy(lower_bound)
                        cur_neuron_concrete_algebra_upper = deepcopy(upper_bound)
                        quantized_concrete_algebra_lower.append(cur_neuron_concrete_algebra_lower)
                        quantized_concrete_algebra_upper.append(cur_neuron_concrete_algebra_upper)

                    # reverse, from cur_layer's affine layer to input layer
                    for kk in range(2 * (in_layer_index + 1) - 1)[::-1]:
                        # size of input
                        tmp_lower = np.zeros(len(self.deepPolyNets_DNN.layers[kk].neurons[0].algebra_lower))
                        tmp_upper = np.zeros(len(self.deepPolyNets_DNN.layers[kk].neurons[0].algebra_lower))

                        assert (self.deepPolyNets_DNN.layers[kk].size + 1 == len(lower_bound))
                        assert (self.deepPolyNets_DNN.layers[kk].size + 1 == len(upper_bound))

                        for pp in range(self.deepPolyNets_DNN.layers[kk].size):
                            if lower_bound[pp] >= 0:
                                tmp_lower += np.float32(
                                    lower_bound[pp] * self.deepPolyNets_DNN.layers[kk].neurons[
                                        pp].algebra_lower)
                            else:
                                tmp_lower += np.float32(
                                    lower_bound[pp] * self.deepPolyNets_DNN.layers[kk].neurons[
                                        pp].algebra_upper)

                            if upper_bound[pp] >= 0:
                                tmp_upper += np.float32(
                                    upper_bound[pp] * self.deepPolyNets_DNN.layers[kk].neurons[
                                        pp].algebra_upper)
                            else:
                                tmp_upper += np.float32(
                                    upper_bound[pp] * self.deepPolyNets_DNN.layers[kk].neurons[
                                        pp].algebra_lower)

                        tmp_lower[-1] += lower_bound[-1]
                        tmp_upper[-1] += upper_bound[-1]
                        lower_bound = deepcopy(tmp_lower)
                        upper_bound = deepcopy(tmp_upper)
                        #
                        if kk == 1:
                            cur_neuron_concrete_algebra_lower = deepcopy(lower_bound)
                            cur_neuron_concrete_algebra_upper = deepcopy(upper_bound)
                            quantized_concrete_algebra_lower.append(cur_neuron_concrete_algebra_lower)
                            quantized_concrete_algebra_upper.append(cur_neuron_concrete_algebra_upper)

                    assert (len(lower_bound) == 1)
                    assert (len(upper_bound) == 1)

                    cur_neuron_concrete_lower = lower_bound[0]
                    cur_neuron_concrete_upper = upper_bound[0]

                    tmp_acc_lower += cur_neuron_concrete_lower
                    tmp_acc_upper += cur_neuron_concrete_upper

                    pre_mul_qu_lb_deepPoly.append(tmp_acc_lower)
                    pre_mul_qu_ub_deepPoly.append(tmp_acc_upper)

                    # generate property constraints
                    # get quantized_ub_expression from backward-procedure
                    quantized_lb_expression = np.dot(cur_neuron_concrete_algebra_lower[:-1],
                                                     self.input_gp_vars)
                    quantized_lb_expression = quantized_lb_expression + cur_neuron_concrete_algebra_lower[
                        -1]

                    quantized_ub_expression = np.dot(cur_neuron_concrete_algebra_upper[:-1],
                                                     self.input_gp_vars)
                    quantized_ub_expression = quantized_ub_expression + cur_neuron_concrete_algebra_upper[
                        -1]

                    # either lower or higher
                    if cur_layer.layer_index == (len(self.dense_layers) + 1):
                        if out_index == self.targetCls:
                            target_lb = quantized_lb_expression
                        else:
                            other_ubs.append(quantized_ub_expression)
                    else:
                        k_i_lb = self.gp_model.addVar(vtype=GRB.BINARY)
                        var_ll.append(k_i_lb)

                        if cur_layer.relaxed_ub[out_index] > 0:
                            prop_cstr_ll.append(self.gp_model.addConstr(
                                quantized_lb_expression <= cur_layer.relaxed_lb_expression[out_index] - 1000 * (
                                        k_i_lb - 1) - self.tole))
                            prop_cstr_ll.append(self.gp_model.addConstr(
                                quantized_lb_expression >= cur_layer.relaxed_lb_expression[
                                    out_index] - 1000 * k_i_lb + self.tole))
                            sumOfK = sumOfK + k_i_lb
                            numOfK += 1

                        # k_i encodes: is not included
                        # for upper bounds
                        k_i_ub = self.gp_model.addVar(vtype=GRB.BINARY)
                        var_ll.append(k_i_ub)
                        prop_cstr_ll.append(self.gp_model.addConstr(
                            quantized_ub_expression >= cur_layer.relaxed_ub_expression[out_index] + 1000 * (
                                    k_i_ub - 1) + self.tole))
                        prop_cstr_ll.append(self.gp_model.addConstr(
                            quantized_ub_expression <= cur_layer.relaxed_ub_expression[
                                out_index] + 1000 * k_i_ub - self.tole))

                        numOfK += 1
                        sumOfK = sumOfK + k_i_ub

                if len(other_ubs) > 0:  # output layer
                    # k_i encodes: is not included
                    # for upper bounds
                    for other_single_ub in other_ubs:
                        k_i_ub = self.gp_model.addVar(vtype=GRB.BINARY)
                        var_ll.append(k_i_ub)
                        prop_cstr_ll.append(self.gp_model.addConstr(
                            other_single_ub >= target_lb + 1000 * (
                                    k_i_ub - 1) + self.tole))
                        prop_cstr_ll.append(self.gp_model.addConstr(
                            other_single_ub <= target_lb + 1000 * k_i_ub - self.tole))

                        sumOfK = sumOfK + k_i_ub
                        numOfK += 1

                # for relaxed version of Quadapter
                if len(other_ubs) == 0 and self.ifRelax == 1:

                    scale = 0.25

                    # # for better performance, can try this relaxation
                    # if self.scaleValueSet[in_layer_index] <= 0.01 and in_layer_index > 0:
                    #     scale = 0.35

                    prop_cstr_ll.append(self.gp_model.addConstr(sumOfK >= int(numOfK * scale) + 1))
                else:
                    prop_cstr_ll.append(self.gp_model.addConstr(sumOfK >= 1))

                self.gp_model.update()
                self.gp_model.setParam('DualReductions', 0)

                self.gp_model.optimize()

                ifgpUNSat = self.gp_model.status == GRB.INFEASIBLE

                if ifgpUNSat:
                    print("We find a quantization configuration [ Q , F ] for the Layer", cur_layer.layer_index,
                          "as: [", all_bit, ",", frac_bit, '].')

                    cur_layer.frac_bit = frac_bit

                    qu_frac_list.append(cur_layer.frac_bit)
                    qu_int_list.append(cur_layer.int_bit)
                    qu_list.append(all_bit)

                    ifFound = True

                    self.gp_model.remove(model_cstr_ll)
                    self.gp_model.remove(prop_cstr_ll)

                    self.gp_model.remove(var_ll)

                    self.gp_model.update()

                    self.update_quantized_weights_affine(in_layer, cur_layer, all_bit, frac_bit, frac_bit,
                                                         in_layer_index)

                    # if hidden layer, then update next relu's algebra for the abstract element cf. DeepPoly
                    if cur_layer.layer_index < (len(self.dense_layers) + 1):
                        for out_index in range(cur_layer.layer_size):
                            lb_new = pre_mul_qu_lb_deepPoly[out_index]
                            ub_new = pre_mul_qu_ub_deepPoly[out_index]
                            cur_neuron = self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1)].neurons[out_index]
                            if lb_new >= 0:
                                cur_neuron.algebra_lower = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_upper = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_lower[out_index] = 1
                                cur_neuron.algebra_upper[out_index] = 1
                            elif ub_new <= 0:
                                cur_neuron.algebra_lower = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_upper = np.zeros(cur_layer.layer_size + 1)
                            elif lb_new + ub_new <= 0:
                                cur_neuron.algebra_lower = np.zeros(cur_layer.layer_size + 1)
                                k_new = ub_new / (ub_new - lb_new)
                                cur_neuron.algebra_upper = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_upper[out_index] = k_new
                                cur_neuron.algebra_upper[-1] = - k_new * lb_new
                            else:
                                cur_neuron.algebra_lower = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_lower[out_index] = 1
                                k_new = ub_new / (ub_new - lb_new)
                                cur_neuron.algebra_upper = np.zeros(cur_layer.layer_size + 1)
                                cur_neuron.algebra_upper[out_index] = k_new
                                cur_neuron.algebra_upper[-1] = - k_new * lb_new
                    else:
                        self.output_layer.qu_lb = pre_mul_qu_lb_deepPoly
                        self.output_layer.qu_ub = pre_mul_qu_ub_deepPoly
            if not ifFound:
                print("Cannot find a quantization strategy for the cur_layer with index as: ", cur_layer.layer_index)
                return False, None, None, None

        return True, qu_list, qu_frac_list, qu_int_list

    # update deepPoly model's quantized weights
    def update_quantized_weights_affine(self, in_layer, out_layer, num_bit, frac_bit_weights, frac_bit_bias,
                                        in_layer_index):
        min_fp_weight, max_fp_weight = int_get_min_max(num_bit, frac_bit_weights)
        min_fp_bias, max_fp_bias = int_get_min_max(num_bit, frac_bit_bias)
        for out_index in range(out_layer.layer_size):
            weight_row = out_layer.layer_paras[0][out_index]
            bias = out_layer.layer_paras[1][out_index]
            weight_row_int = quantize_int(np.asarray(weight_row), num_bit, frac_bit_weights)
            weight_row_fp = np.clip(weight_row_int / (2 ** frac_bit_weights),
                                    min_fp_weight, max_fp_weight)
            weight_row_fp = weight_row_int / (2 ** frac_bit_weights)

            bias_fp = quantize_int(bias, num_bit, frac_bit_bias) / (2 ** frac_bit_bias)

            # update weight and bias parameters for affine layer
            self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[out_index].weight = weight_row_fp
            self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[out_index].bias = bias_fp

            # update algebra parameters for affine layer
            self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[out_index].algebra_lower = np.append(
                weight_row_fp, [bias_fp])
            self.deepPolyNets_DNN.layers[2 * (in_layer_index + 1) - 1].neurons[out_index].algebra_upper = np.append(
                weight_row_fp, [bias_fp])

    def write_result(self, qu_frac_list, fileName):
        real_qu_list = []
        frac_qu_list = []
        int_qu_list = []
        for i, l in enumerate(self.dense_layers):
            real_qu = qu_frac_list[i] + self.dense_layers[i].int_bit
            real_qu_list.append(real_qu)
            frac_qu_list.append(qu_frac_list[i])
            int_qu_list.append(self.dense_layers[i].int_bit)

        real_qu_list.append(qu_frac_list[-1] + self.output_layer.int_bit)
        frac_qu_list.append(qu_frac_list[-1])
        int_qu_list.append(self.output_layer.int_bit)

        print("\n******************** The quantization strategy holds the property ********************\n")
        fo = open(fileName, "w")
        fo.write("Solving Result: True\n")
        fo.write("We found a quantization strategy to hold the robustness property.\n")
        fo.write("The all quantization bit sizes for each layer are:" + str(real_qu_list) + "\n")
        fo.write("The frac quantization bit sizes for each layer are:" + str(frac_qu_list) + "\n")
        fo.write("The int quantization bit sizes for each layer are:" + str(int_qu_list) + "\n")
        fo.write("Backward Time: " + str(self._stats["backward_time"]) + "\n")
        fo.write("Forward Time: " + str(self._stats["forward_time"]) + "\n")
        fo.write("Total Time: " + str(self._stats["total_time"]) + "\n")
        numAllVars = self.gp_model.getAttr("NumVars")
        numIntVars = self.gp_model.getAttr("NumIntVars")
        numBinVars = self.gp_model.getAttr("NumBinVars")
        numConstrs = self.gp_model.getAttr("NumConstrs")
        #
        fo.write("The num of vars: " + str(numAllVars) + "\n")
        fo.write("The num of numIntVars: " + str(numIntVars) + "\n")
        fo.write("The num of numBinVars: " + str(numBinVars) + "\n")
        fo.write("The num of Constraints: " + str(numConstrs) + "\n")
        fo.close()