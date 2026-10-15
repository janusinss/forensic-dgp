"""Fresh maintenance-only inventory after the manual A return; no deletion."""
import cctv_dgp_bank_archive_cleanup_v1 as inventory


if __name__ == '__main__':
    inventory.OUT = inventory.ROOT / 'outputs/cctv_dgp_bank_A_storage_inventory_v1'
    inventory.transport.OUT = inventory.OUT
    inventory.inventory()
