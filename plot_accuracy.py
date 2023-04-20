import matplotlib.pyplot as plt
import numpy as np

def extract_test_accs_from_log(logfile):
    test_accs = []
    with open(logfile, 'r') as f:
        lines = f.readlines()
    for l in lines:
        if 'accuracy' in l:
            test_accs.append(100.0*float(l.split(',')[-1]))
    return test_accs

directory = "lottery_108ebe7f8e2dbe540dbed0e9011edec1/"
full_model_log = directory + "replicate_1/level_0/main/logger"
IMP_log = directory + "replicate_1/level_10/main/logger"
even_split_log = directory + "replicate_120/level_10/lottery_branch_retrain_cd11e6b3762f61c5b08d1c241f590336/logger"
# uneven_split_onefourth_log = directory + "replicate_101/level_10/lottery_branch_retrain_cd11e6b3762f61c5b08d1c241f590336/logger"
# uneven_split_oneeighth_log = directory + "replicate_102/level_10/lottery_branch_retrain_cd11e6b3762f61c5b08d1c241f590336/logger"
# uneven_split_zero_log = directory + "replicate_103/level_10/lottery_branch_retrain_cd11e6b3762f61c5b08d1c241f590336/logger"
perturbed_even_split_log = directory + "replicate_121/level_10/lottery_branch_retrain_cd11e6b3762f61c5b08d1c241f590336/logger"

full_model_accuracy = extract_test_accs_from_log(full_model_log)
IMP_accuracy = extract_test_accs_from_log(IMP_log)
even_split_accuracy = extract_test_accs_from_log(even_split_log)
perturbed_even_split_accuracy = extract_test_accs_from_log(perturbed_even_split_log)

# uneven_split_onefourth_accuracy = extract_test_accs_from_log(uneven_split_onefourth_log)
# uneven_split_oneeighth_accuracy = extract_test_accs_from_log(uneven_split_oneeighth_log)
# uneven_split_zero_accuracy = extract_test_accs_from_log(uneven_split_zero_log)

print("The max accuracy achieved by full model is ", np.max(full_model_accuracy))
print("The max accuracy achieved by IMP with sparsity 89.26 is ", np.max(IMP_accuracy))
print("The max accuracy achieved by even split is ", np.max(even_split_accuracy))
print("The max accuracy achieved by perturbed even split is ", np.max(perturbed_even_split_accuracy))
# print("The max accuracy achieved by uneven split (1/4, 3/4) is ", np.max(uneven_split_onefourth_accuracy))
# print("The max accuracy achieved by uneven split (1/8, 7/8) is ", np.max(uneven_split_oneeighth_accuracy))
# print("The max accuracy achieved by uneven split (0) is ", np.max(uneven_split_zero_accuracy))

plt.plot(full_model_accuracy[1:], label="full model")
plt.plot(IMP_accuracy[1:], label="IMP")
plt.plot(even_split_accuracy[1:], label="even split")
plt.plot(perturbed_even_split_accuracy[1:], label="perturbed even split")
# plt.plot(uneven_split_onefourth_accuracy[1:], label="uneven split (1/4,3/4)")
# plt.plot(uneven_split_oneeighth_accuracy[1:], label="uneven split (1/8,7/8)")
# plt.plot(uneven_split_zero_accuracy[1:], label="uneven split (0,1)")
plt.title("MNIST_Lenet_300_100")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()