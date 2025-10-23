import gc
import jax
import jax.numpy as jnp


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flatten a nested dictionary into a flat dictionary.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The parent key for the current level of the dictionary.
        sep (str): The separator to use between keys.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def reset_device_memory(delete_live_buffers: bool = True) -> int:
    """Best-effort device memory reset for modern JAX.

    Returns:
      int: number of backend live buffers/arrays explicitly deleted (best-effort).
    """
    n_deleted = 0

    # 1) Drop Python references to jax.Array objects and collect.
    #    (We don't rely on private types like DeviceArray/DeviceValue anymore.)
    try:
        for obj in gc.get_objects():
            # jax.Array is the public runtime type in modern JAX
            if isinstance(obj, jax.Array):
                # Ensure pending transfers/compute finish (avoids racing with deletion)
                try:
                    obj.block_until_ready()
                except Exception:
                    pass
                # Let GC reclaim host refs; device buffers go when last ref is gone
                # (can't del 'obj' here meaningfully; just hint GC)
        # A couple of collect passes helps in practice
        gc.collect()
        gc.collect()
    except Exception:
        # If gc.get_objects() is restricted (PyPy, etc.), just proceed
        pass

    # 2) Clear JAX caches that may pin executables / device allocations.
    try:
        jax.clear_caches()
    except Exception:
        pass

    # 3) Best-effort explicit deletion of backend live buffers/arrays (private-ish).
    if delete_live_buffers:
        try:
            backend = jax.lib.xla_bridge.get_backend()
        except Exception:
            backend = None

        if backend is not None:
            # Older paths: live_buffers(); some PJRT builds: live_arrays()
            deleted_here = 0
            for attr in ("live_buffers", "live_arrays"):
                try:
                    live = getattr(backend, attr)()
                except Exception:
                    continue
                # Materialize list in case the iterator is view-like
                for buf in list(live):
                    try:
                        buf.delete()   # PJRT Buffer/Array has delete()
                        deleted_here += 1
                    except Exception:
                        pass
                if deleted_here:
                    n_deleted += deleted_here
                    break  # no need to try the other attribute

    # One more GC pass after backend deletes
    gc.collect()
    return n_deleted
def stable_mean(x):
    # Create a mask where `True` indicates non-NaN values
    nan_check = ~jnp.isnan(x)
    inf_check = ~jnp.isinf(x)
    mask = nan_check & inf_check

    # Replace NaNs with zero
    x = jnp.where(mask, x, 0)

    # Compute the sum of non-NaN values
    total_sum = jnp.sum(x)

    # Compute the number of non-NaN values
    count = jnp.sum(mask)

    # Compute the mean of non-NaN values
    mean_value = total_sum / count

    return mean_value


def replace_invalid(x, replacement=0.):
    # Create a mask where `True` indicates non-NaN values
    nan_check = ~jnp.isnan(x)
    inf_check = ~jnp.isinf(x)
    mask = nan_check & inf_check

    # Replace NaNs with zero
    x = jnp.where(mask, x, replacement)

    # Compute the number of non-NaN values
    invalid_count = jnp.sum(mask)

    return x, invalid_count


def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1)

def flattened_traversal(fn):
    def mask(data):
        flat = traverse_util.flatten_dict(data)
        return traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})

    return mask


if __name__ == '__main__':
    init_std = 10
    a = inverse_softplus(10)
    print(jax.nn.softplus(a))
