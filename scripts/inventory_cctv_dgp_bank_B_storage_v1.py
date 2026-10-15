"""Fresh maintenance-only inventory after the manually returned B load stop."""
import cctv_dgp_bank_archive_cleanup_v1 as inventory


if __name__ == '__main__':
    inventory.OUT = inventory.ROOT / 'outputs/cctv_dgp_bank_B_storage_inventory_v1'
    inventory.transport.OUT = inventory.OUT
    inventory.inventory()
