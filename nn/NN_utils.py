import jax, numpy as jnp

def get_bias_flat2paratree(tree):

    tree_flat, tree_flat_fu = jax.flatten_util.ravel_pytree(tree)
    wb_len, b_true = [], []
    count = 0
    ele_pre = 1
    for ele in tree_flat:
        if (ele == 1 and ele_pre == 1):
            count += 1
            ele_pre = 1
        elif (ele == 0 and ele_pre == 1):
            wb_len.append(count)
            b_true.append(1)
            count = 1
            ele_pre = 0
        elif (ele == 0 and ele_pre == 0):
            count += 1
            ele_pre = 0
        elif (ele == 1 and ele_pre == 0):
            wb_len.append(count)
            b_true.append(0)
            count = 1
            ele_pre = 1

    if ele == 1:
        wb_len.append(count)
        b_true.append(1)
    elif ele == 0:
        wb_len.append(count)
        b_true.append(0)

    nbias = sum([l*bt for l, bt in zip(wb_len, b_true)])

    def bias_flat2paratree(tree_flat_bias):
        array = []
        idx_trac = 0
        for l, bt in zip(wb_len, b_true):
            if bt:
                arr_ = tree_flat_bias[idx_trac:idx_trac+l]
            else:
                arr_ = jnp.zeros(l)
            array.append(arr_)
            idx_trac += l*bt
        
        return tree_flat_fu(jnp.concatenate(array))
    
    return bias_flat2paratree, nbias
    

def get_flat2paratree(tree):

    tot_para = 0
    for w in jax.tree_util.tree_leaves(tree):
        tot_para += w.size
    
    tree_flat, tree_flat_fu = jax.flatten_util.ravel_pytree(tree)
    def flat2paratree(tree_flat):
        return tree_flat_fu(tree_flat)
    
    return flat2paratree, tot_para
    