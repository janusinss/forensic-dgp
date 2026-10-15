"""Fresh read-only inventory for the already authorized VM storage maintenance."""
import cctv_dgp_bank_archive_cleanup_v1 as inventory


if __name__ == '__main__':
    inventory.OUT = inventory.ROOT / 'outputs/cctv_dgp_bank_group_gradients_v1_storage_cleanup'
    inventory.transport.OUT = inventory.OUT
    inventory.inventory()
